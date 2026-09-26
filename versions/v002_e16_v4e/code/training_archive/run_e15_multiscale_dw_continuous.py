"""E15: epoch170 -> epoch200, multi-scale depthwise branch and new-parameter adaptation.

Uses the audited E14B name-based migration for the eight historical MuSGD
groups, then isolates the five new parameters into five additional groups.
Training requires --train; the default only prepares and checks the model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/matplotlib"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model
from src.models import register_custom_modules
from src.models.multiscale_dw_branch import TARGET, MultiScaleBottleneck, attach_multiscale_branch
from scripts.run_e14b_dysample_continuous import (
    canonical_musgd_group_names, continuation_lr_factor, parameter_shapes,
    remap_optimizer_state_dict, sha256_file, _state_tensor_audit,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/E15_multiscale_dw_continuous.yaml"
NEW_NAMES = tuple(TARGET + ".ms_branch." + n for n in (
    "project.weight", "reduce.weight", "scales.0.weight", "scales.1.weight", "scales.2.weight"))


def resolve(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def snapshot(model):
    result = deepcopy(unwrap_model(model)).float().cpu()
    if hasattr(result, "criterion"):
        result.criterion = None
    for name, value in result.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite checkpoint tensor: {name}")
    return result


def experiment_contract(config):
    files = [Path(__file__), DEFAULT_CONFIG, resolve(config["model_yaml"]),
             ROOT / "src/models/multiscale_dw_branch.py",
             ROOT / "scripts/run_e14b_dysample_continuous.py", resolve(config["data_yaml"])]
    contract = {"experiment_id": "E15", "config": config,
                "files": {str(p.relative_to(ROOT)): sha256_file(p) for p in files}}
    contract["fingerprint"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return contract


def check_architecture(model):
    modules = dict(unwrap_model(model).named_modules())
    if [n for n,m in modules.items() if isinstance(m, MultiScaleBottleneck)] != [TARGET]:
        raise RuntimeError("E15 requires exactly one branch at " + TARGET)
    for i in (11, 14):
        m = unwrap_model(model).model[i]
        if not isinstance(m, torch.nn.Upsample) or m.mode != "nearest":
            raise RuntimeError("Original nearest upsampling must remain")
    if list(model.stride.cpu().tolist()) != [8.,16.,32.]:
        raise RuntimeError("Detection strides changed")


def split_new_groups(optimizer, model, multiplier):
    """Split existing groups without recreating parameter objects or old state."""
    if len(optimizer.param_groups) != 8:
        raise RuntimeError("Expected eight original groups before splitting")
    named = dict(unwrap_model(model).named_parameters())
    for name in NEW_NAMES:
        parameter = named[name]
        candidates = [g for g in optimizer.param_groups[:8] if any(p is parameter for p in g["params"])]
        if len(candidates) != 1:
            raise RuntimeError(f"New parameter grouping failed: {name}")
        original = candidates[0]
        new_group = {k: deepcopy(v) for k,v in original.items() if k != "params"}
        original["params"] = [p for p in original["params"] if p is not parameter]
        new_group.update(params=[parameter], lr=original["lr"]*multiplier,
                         initial_lr=original.get("initial_lr",original["lr"])*multiplier,
                         e15_new=True)
        optimizer.add_param_group(new_group)
    return optimizer_group_names(optimizer, model)


def optimizer_group_names(optimizer, model):
    names = {id(p):n for n,p in unwrap_model(model).named_parameters()}
    return [[names[id(p)] for p in g["params"]] for g in optimizer.param_groups]


def new_warmup_factor(updates, total=250, start=.1):
    return start + (1-start)*min(1., (updates+1)/total)


class SeparateAgeEMA(ModelEMA):
    @classmethod
    def wrap(cls, existing, new_updates, config):
        obj = cls.__new__(cls)
        obj.__dict__.update(existing.__dict__)
        obj.new_updates = int(new_updates)
        obj.new_tau = float(config["new_ema_tau"])
        obj.max_decay = float(config["ema_decay"])
        return obj

    def update(self, model):
        if not self.enabled:
            return
        self.updates += 1
        self.new_updates += 1
        old_decay = self.decay(self.updates)
        new_decay = self.max_decay*(1-math.exp(-self.new_updates/self.new_tau))
        with torch.no_grad():
            live = unwrap_model(model).state_dict()
            for name,value in self.ema.state_dict().items():
                if value.is_floating_point():
                    decay = new_decay if name in NEW_NAMES else old_decay
                    value.mul_(decay).add_(live[name].detach(), alpha=1-decay)


def prepare(config, contract):
    source_path = resolve(config["source_checkpoint"])
    if sha256_file(source_path) != config["source_sha256"]:
        raise RuntimeError("Source epoch170 checkpoint hash mismatch")
    artifact = resolve(config["artifacts"])
    prepared = artifact / "prepared" / f"epoch170_multiscale_{contract['fingerprint'][:12]}.pt"
    if prepared.is_file():
        return prepared
    register_custom_modules()
    torch.manual_seed(config["seed"])
    source, ckpt = load_checkpoint(source_path, device="cpu")
    source = source.float().eval()
    if ckpt["epoch"] != 169 or ckpt.get("optimizer") is None or ckpt.get("scaler") is None:
        raise RuntimeError("Source must be completed epoch170 with optimizer and AMP state")
    if ckpt["updates"] != 43017 or len(ckpt["optimizer"]["state"]) != 330:
        raise RuntimeError("Source state does not match the audited epoch170 base")
    model = attach_multiscale_branch(source).float().eval()
    model.names = deepcopy(source.names)
    check_architecture(model)
    source_sd, target_sd = source.state_dict(), model.state_dict()
    if not all(k in target_sd and torch.equal(v,target_sd[k]) for k,v in source_sd.items()):
        raise RuntimeError("Original model tensors were not fully preserved")
    new_params = sorted(set(dict(model.named_parameters()))-set(dict(source.named_parameters())))
    if new_params != list(NEW_NAMES):
        raise RuntimeError(f"Unexpected new parameters: {new_params}")
    checks = {}
    with torch.inference_mode():
        x = torch.rand(1,3,256,320)
        y0,y1 = source(x)[0],model(x)[0]
        checks["cpu_fp32_initial_max_abs_error"] = float((y0-y1).abs().max())
        if not torch.equal(y0,y1):
            raise RuntimeError("Initial residual model is not identical to the base")
    args = deepcopy(ckpt["train_args"])
    args.update(model=str(resolve(config["model_yaml"])), data=str(resolve(config["data_yaml"])),
                epochs=200, optimizer="MuSGD", warmup_epochs=0., val=False, plots=False,
                close_mosaic=50, project=str(ROOT/"runs/yolo11"), name=config["run_name"],
                save_dir=str(ROOT/"runs/yolo11"/config["run_name"]))
    model.args = deepcopy(args)
    payload = dict(ckpt)
    payload.update(model=None, ema=snapshot(model), train_args=args,
                   e15={"kind":"prepared", "contract":contract,
                          "source_group_names":canonical_musgd_group_names(source),
                          "source_shapes":parameter_shapes(source), "new_updates":0})
    save_checkpoint(prepared,payload)
    write_json(artifact/"preparation_report.json", {
        "prepared":str(prepared),"source":str(source_path),"source_sha256":config["source_sha256"],
        "shared_state_items":len(source_sd),"new_parameters":new_params,
        "new_parameter_count":sum(dict(model.named_parameters())[n].numel() for n in NEW_NAMES),
        "checks":checks,"contract":contract})
    return prepared


class E15Trainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        # The saved base YAML has no branch: attach BEFORE loading custom weights.
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        model = attach_multiscale_branch(model)
        if weights is not None:
            model.load(weights)
        check_architecture(model)
        return model

    def __init__(self, *args, config, contract, run_dir, smoke=False, **kwargs):
        self.experiment = config
        self.contract = contract
        self.run_dir = run_dir
        self.smoke = smoke
        self.stop_epoch = 171 if smoke else config["stop_epoch"]
        self.batch_counter = 0
        self.start_time = time.time()
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_batch_end", self.progress_callback)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if not self.resume:
            raise RuntimeError("E15 requires an audited checkpoint")
        c,a = self.experiment,self.args
        a.epochs=self.stop_epoch
        a.data=str(resolve(c["data_yaml"]))
        a.project=str(self.run_dir.parent)
        a.name=self.run_dir.name
        a.save_dir=str(self.run_dir)
        a.exist_ok=True
        a.imgsz=c["imgsz"]
        a.batch=1 if self.smoke else c["batch"]
        a.workers=0 if self.smoke else c["workers"]
        a.device=c["device"]
        a.seed=c["seed"]
        a.optimizer="MuSGD"
        a.warmup_epochs=0.
        a.close_mosaic=self.stop_epoch-150
        a.val=False
        a.plots=False
        a.compile=False
        a.amp=True
        a.save=True
        a.save_period=-1
        a.fraction=c["smoke_fraction"] if self.smoke else 1.
        a.time=None
        a.freeze=None
        a.patience=0

    def _build_train_pipeline(self):
        if getattr(self,"pipeline_built",False):
            raise RuntimeError("E15 stopped on OOM; automatic optimizer rebuilding is disabled")
        self.pipeline_built=True
        return super()._build_train_pipeline()

    def build_optimizer(self, model, name="auto", lr=.001, momentum=.9, decay=1e-5, iterations=1e5):
        result=super().build_optimizer(model,"MuSGD",lr,momentum,decay,iterations)
        if len(result.param_groups)!=8:
            raise RuntimeError("Unexpected MuSGD base layout")
        return result

    def _load_checkpoint_state(self, ckpt):
        meta=ckpt.get("e15") or {}
        if meta.get("contract",{}).get("fingerprint")!=self.contract["fingerprint"]:
            raise RuntimeError("Checkpoint/config/code contract mismatch")
        prepared=meta.get("kind")=="prepared"
        if prepared:
            state,audit=remap_optimizer_state_dict(
                ckpt["optimizer"],meta["source_group_names"],meta["source_shapes"],
                self.optimizer,canonical_musgd_group_names(self.model),self.model,NEW_NAMES)
            modified=dict(ckpt,optimizer=state)
            super()._load_checkpoint_state(modified)
            split_new_groups(self.optimizer,self.model,self.experiment["new_lr_multiplier"])
        else:
            if meta.get("run_name")!=self.run_dir.name or meta.get("kind")!="trained":
                raise RuntimeError("Checkpoint belongs to another run")
            names=split_new_groups(self.optimizer,self.model,self.experiment["new_lr_multiplier"])
            if names!=meta.get("group_names"):
                raise RuntimeError("Native optimizer parameter order changed")
            super()._load_checkpoint_state(ckpt)
            if ckpt.get("model") is None:
                raise RuntimeError("E15 native resume requires its live model snapshot")
            unwrap_model(self.model).load_state_dict(ckpt["model"].float().state_dict(),strict=True)
            audit={"restore_kind":"native_live_and_ema"}
        self.ema=SeparateAgeEMA.wrap(self.ema,meta["new_updates"],self.experiment)
        check_architecture(unwrap_model(self.model))
        audit.update(_state_tensor_audit(self.optimizer))
        expected=330 if prepared else 335
        if audit["optimizer_state_count"]!=expected or not audit["all_state_tensors_finite"]:
            raise RuntimeError("Optimizer state audit failed")
        if self.scaler.state_dict()!=ckpt["scaler"]:
            raise RuntimeError("AMP scaler migration failed")
        self.restore_audit=audit

    def lr_factor(self, epoch):
        c=self.experiment
        return continuation_lr_factor(epoch,c["scheduler_anchor_epoch"],
            c["scheduler_horizon_epoch"],c["scheduler_final_factor"])

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        if not 170<=self.start_epoch<self.stop_epoch:
            raise RuntimeError("Unexpected continuation epoch")
        lrs=[float(g["lr"]) for g in self.optimizer.param_groups]
        for i,g in enumerate(self.optimizer.param_groups[:8]):
            if not math.isclose(g["lr"],g["initial_lr"]*self.lr_factor(ckpt["epoch"]),rel_tol=5e-8):
                raise RuntimeError(f"Historical learning-rate phase changed in group {i}")
        self.lf=self.lr_factor
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=self.lf)
        for g,lr in zip(self.optimizer.param_groups,lrs):
            g["lr"]=lr
        self.scheduler.last_epoch=self.start_epoch-1
        self.scheduler._last_lr=lrs
        self._close_dataloader_mosaic()
        self.metrics={}
        self.fitness=None
        self.best_fitness=None
        report=dict(self.restore_audit, next_epoch=self.start_epoch+1,stop_epoch=self.stop_epoch,
                    ema_updates=self.ema.updates,new_updates=self.ema.new_updates,
                    group_names=optimizer_group_names(self.optimizer,self.model),
                    loaded_lrs=lrs,mosaic_closed=True,scaler=self.scaler.state_dict())
        write_json(self.run_dir/"state_restore_report.json",report)

    def optimizer_step(self):
        c=self.experiment
        warm=new_warmup_factor(self.ema.new_updates,c["new_warmup_updates"],c["new_warmup_start_factor"])
        for g in self.optimizer.param_groups[8:]:
            g["lr"]=g["initial_lr"]*self.lf(self.epoch)*warm
        self.scaler.unscale_(self.optimizer)
        grads=[dict(self.model.named_parameters())[n].grad for n in NEW_NAMES]
        self.last_new_gradient_norm=float(torch.stack([g.detach().float().norm() for g in grads if g is not None]).norm())
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.)
        before=self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        stepped=self.scaler.get_scale()>=before
        self.optimizer.zero_grad()
        if stepped:
            self.ema.update(self.model)

    def progress_callback(self, trainer):
        self.batch_counter+=1
        if self.batch_counter==1 or self.batch_counter%50==0:
            write_json(self.run_dir/"progress.json",{
                "status":"running","pid":os.getpid(),"updated_at":datetime.now().isoformat(),
                "epoch":self.epoch+1,"target_epoch":self.stop_epoch,
                "completed_epoch":self.epoch,"batches_this_process":self.batch_counter,
                "elapsed_seconds":time.time()-self.start_time,
                "loss":self.tloss.detach().float().cpu().tolist(),
                "new_updates":self.ema.new_updates,"ema_updates":self.ema.updates,
                "new_parameter_lrs":[g["lr"] for g in self.optimizer.param_groups[8:]],
                "last_new_gradient_norm":getattr(self,"last_new_gradient_norm",None)})

    def validate(self):
        return {},None

    def final_eval(self):
        return None

    def _handle_nan_recovery(self, epoch):
        if not torch.isfinite(self.loss).all():
            raise RuntimeError("Nonfinite loss; stopped without changing experiment state")
        return False

    def save_model(self):
        c=self.experiment
        audit=_state_tensor_audit(self.optimizer)
        if audit["optimizer_state_count"]!=335 or not audit["all_state_tensors_finite"]:
            raise RuntimeError("New parameters did not acquire finite optimizer state")
        live,ema=snapshot(self.model),snapshot(self.ema.ema)
        meta={"kind":"trained","contract":self.contract,"run_name":self.run_dir.name,
              "new_updates":self.ema.new_updates,
              "group_names":optimizer_group_names(self.optimizer,self.model)}
        payload={"epoch":self.epoch,"best_fitness":None,"model":live,"ema":ema,
                 "updates":self.ema.updates,"optimizer":cpu_tree(self.optimizer.state_dict()),
                 "scaler":self.scaler.state_dict(),"train_args":vars(self.args),
                 "train_metrics":{},"train_results":self.read_results_csv(),
                 "date":datetime.now().isoformat(),"version":ultralytics.__version__,"e15":meta}
        save_checkpoint(self.last,payload)
        epoch=self.epoch+1
        if epoch in c["save_epochs"] or epoch==self.stop_epoch:
            destination=self.wdir/f"epoch{epoch}.pt"
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(self.last,destination)
            live_payload={k:v for k,v in payload.items() if k not in {"ema","optimizer","e15"}}
            live_payload.update(ema=None,optimizer=None,epoch=-1,
                                e15_inference={"completed_epoch":epoch,"weights":"live"})
            save_checkpoint(self.wdir/f"epoch{epoch}_live.pt",live_payload)
        write_json(self.run_dir/"latest_state_audit.json",dict(audit,
            completed_epoch=epoch,new_updates=self.ema.new_updates,ema_updates=self.ema.updates,
            new_weight_std=float(live.get_submodule(TARGET).ms_branch.project.weight.detach().std()),
            ema_new_weight_std=float(ema.get_submodule(TARGET).ms_branch.project.weight.detach().std()),
            checkpoint=str(self.last)))
        return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--train",action="store_true")
    modes.add_argument("--smoke-test",action="store_true")
    modes.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args(argv)
    config=yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if ultralytics.__version__!="8.4.98" or config["source_completed_epoch"]!=170 or config["stop_epoch"]!=200:
        raise RuntimeError("Unsupported runtime or training budget")
    data=yaml.safe_load(resolve(config["data_yaml"]).read_text(encoding="utf-8"))
    if not data.get("validation_is_train") or config["validation_enabled"]:
        raise RuntimeError("E15 expects disabled validation on the existing all-train dataset")
    torch.set_num_threads(4)
    contract=experiment_contract(config)
    prepared=prepare(config,contract)
    if not args.train and not args.smoke_test:
        print(json.dumps({"status":"prepared","checkpoint":str(prepared)},ensure_ascii=False))
        return
    run_name=(f"smoke_e15_{contract['fingerprint'][:8]}" if args.smoke_test else config["run_name"])
    run_dir=ROOT/"runs/yolo11"/run_name
    if run_dir.exists():
        old=run_dir/"run_contract.json"
        if not old.is_file() or json.loads(old.read_text(encoding="utf-8"))["fingerprint"]!=contract["fingerprint"]:
            raise RuntimeError("Existing run directory has a different contract")
    run_dir.mkdir(parents=True,exist_ok=True)
    lock=run_dir/"training.lock"
    handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(handle,"w",encoding="utf-8") as stream:
        json.dump({"pid":os.getpid(),"created_at":datetime.now().isoformat()},stream)
    write_json(run_dir/"run_contract.json",contract)
    try:
        last=run_dir/"weights/last.pt"
        resume=last if last.is_file() else prepared
        stop=171 if args.smoke_test else 200
        if last.is_file():
            saved=torch_load(last,map_location="cpu")
            complete=saved["epoch"]+1
            del saved
            if complete>=stop:
                print(f"Already completed epoch{complete}: {last}")
                return
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        register_custom_modules()
        yolo=YOLO(str(resume))
        def trainer_factory(*a,**kw):
            return E15Trainer(*a,config=config,contract=contract,run_dir=run_dir,smoke=args.smoke_test,**kw)
        yolo.train(trainer=trainer_factory,resume=str(resume),device=config["device"],
                   imgsz=config["imgsz"],batch=1 if args.smoke_test else config["batch"],
                   workers=0 if args.smoke_test else config["workers"],val=False,plots=False)
        final=torch_load(last,map_location="cpu")
        if final["epoch"]+1!=stop or not (last.parent/f"epoch{stop}.pt").is_file():
            raise RuntimeError("Training exited before the requested endpoint")
        del final
        if sha256_file(resolve(config["source_checkpoint"]))!=config["source_sha256"]:
            raise RuntimeError("Source checkpoint changed")
        report={"status":"complete","completed_epoch":stop,"additional_epochs":stop-170,
                "checkpoint":str(last.parent/f"epoch{stop}.pt"),
                "live_checkpoint":str(last.parent/f"epoch{stop}_live.pt"),
                "finished_at":datetime.now().isoformat(),"source_unchanged":True}
        write_json(run_dir/"completion_report.json",report)
        write_json(run_dir/"progress.json",report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except BaseException as exc:
        write_json(run_dir/"failure_report.json",{"status":"failed","error":repr(exc),
            "at":datetime.now().isoformat(),"pid":os.getpid()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=="__main__":
    main()
