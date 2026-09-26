"""Train a split-isolated detector; V4E is an inference strategy, not a loss."""
import _bootstrap  # noqa
import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import unwrap_model
from common import CODE, VERSION, NAMES, settings, validate_split, sha256, digest, write_json, source_fingerprint, run_lock
from prepare_data import check_ready

def architecture_check(model,kind):
    model=unwrap_model(model)
    branches=[n for n,_ in model.named_modules() if n.endswith('ms_branch')]
    if kind=='e16':
        from src.models.multiscale_dw_output import check_architecture
        check_architecture(model)
    elif branches or any('DySample' in type(m).__name__ for m in model.modules()):
        raise ValueError('Baseline contains a custom branch')
    if model.stride.tolist()!=[8.,16.,32.] or model.model[-1].nc!=9: raise ValueError('Wrong strides/classes')

def build_model(cfg,weights,kind,verbose=False):
    model=DetectionModel(cfg,nc=9,ch=3,verbose=verbose)
    if kind=='e16':
        from src.models.multiscale_dw_output import attach_multiscale_branch
        model=attach_multiscale_branch(model)
    model.names=dict(enumerate(NAMES))
    if weights is not None: model.load(weights)
    architecture_check(model,kind)
    return model

def cpu_tree(x):
    if isinstance(x,torch.Tensor): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [cpu_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(cpu_tree(v) for v in x)
    return x

def snapshot(model):
    m=deepcopy(unwrap_model(model)).float().cpu()
    if hasattr(m,'criterion'): m.criterion=None
    if any(not torch.isfinite(v).all() for v in m.state_dict().values()): raise RuntimeError('Nonfinite weights')
    return m

def atomic_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(payload,tmp);os.replace(tmp,path)

def optimizer_names(optimizer,model):
    names={id(p):n for n,p in model.named_parameters()}
    return [[names[id(p)] for p in g['params']] for g in optimizer.param_groups]

class SplitTrainer(DetectionTrainer):
    def __init__(self,*args,contract,kind,**kwargs):
        self.contract=contract;self.kind=kind
        super().__init__(*args,**kwargs)
        self.wdir=VERSION/'weights';self.wdir.mkdir(exist_ok=True)
        self.last=self.wdir/'last.pt';self.best=self.wdir/'best.pt'
        # The upstream constructor creates an empty nested weights folder.
        nested=self.save_dir/'weights'
        if nested.is_dir() and not any(nested.iterdir()): nested.rmdir()
        self.add_callback('on_train_epoch_end',self.record_progress)
        self.add_callback('on_train_start',self.restore_rng)

    def check_resume(self,overrides):
        super().check_resume(overrides)
        # A relocated checkpoint may remember an existing path on the original computer.
        self.args.data = overrides['data']
        self.args.project=str(CODE/'training_records');self.args.name='train'
        self.args.save_dir=str(CODE/'training_records/train');self.args.exist_ok=True
        if self.resume: self.args.pretrained=True

    def get_model(self,cfg=None,weights=None,verbose=True):
        if weights is None: raise RuntimeError('Official pretrained or resumable weights required')
        model=build_model(cfg,weights,self.kind,verbose)
        if not self.resume:
            if self.kind=='e16' and model.model[16].ms_branch.project.weight.count_nonzero():
                raise RuntimeError('Expected zero-initialized E16 projection')
            write_json(self.save_dir/'initialization.json',dict(architecture=self.kind,official_pretrained=True,
                        transferred_items=sum(k in model.state_dict() and v.shape==model.state_dict()[k].shape for k,v in weights.state_dict().items())))
        return model

    def _load_checkpoint_state(self,ckpt):
        meta=ckpt.get('split_experiment',{})
        if meta.get('contract')!=self.contract or meta.get('architecture')!=self.kind:
            raise RuntimeError('Checkpoint is not from this split/version/configuration')
        if meta['optimizer_names']!=optimizer_names(self.optimizer,self.model): raise RuntimeError('Optimizer ordering changed')
        super()._load_checkpoint_state(ckpt)
        unwrap_model(self.model).load_state_dict(ckpt['model'].float().state_dict(),strict=True)
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.pending_rng=ckpt.get('rng')
        write_json(self.save_dir/'resume.json',dict(completed_epoch=ckpt['epoch']+1,live_model=True,optimizer=True,ema=True,scaler=True,
                    note='Epoch-boundary resume; worker augmentation sequence is not guaranteed bitwise identical.'))

    def restore_rng(self,trainer):
        rng=getattr(self,'pending_rng',None)
        if rng:
            random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
            if torch.cuda.is_available() and rng['cuda'] is not None: torch.cuda.set_rng_state_all(rng['cuda'])

    def save_model(self):
        live=snapshot(self.model);ema=snapshot(self.ema.ema)
        payload=dict(epoch=self.epoch,model=live,ema=ema,updates=self.ema.updates,
                     optimizer=cpu_tree(self.optimizer.state_dict()),scaler=self.scaler.state_dict(),
                     scheduler=deepcopy(self.scheduler.state_dict()),train_args=vars(self.args),
                     train_metrics={**(self.metrics or {}),'fitness':self.fitness},best_fitness=self.best_fitness,
                     train_results=self.read_results_csv(),version=ultralytics.__version__,date=datetime.now().isoformat(),
                     rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                     split_experiment=dict(contract=self.contract,architecture=self.kind,
                              optimizer_names=optimizer_names(self.optimizer,self.model)))
        atomic_save(self.last,payload)
        if self.best_fitness==self.fitness: atomic_save(self.best,payload)
        epoch=self.epoch+1
        if epoch%self.save_period==0 or epoch==self.epochs: atomic_save(self.wdir/f'epoch{epoch}.pt',payload)
        write_json(self.save_dir/'checkpoint_status.json',dict(completed_epoch=epoch,last=str(self.last),best=str(self.best)))
        return True

    def final_eval(self):
        # Preserve live model/optimizer in all saved checkpoints; no strip_optimizer.
        write_json(self.save_dir/'completion.json',dict(completed_epoch=self.epoch+1,best_fitness=self.best_fitness,
                 best_selection='Ultralytics whole-image validation fitness; run evaluate.py for V4E metrics',
                 status='complete' if self.epoch+1>=self.epochs else 'stopped_early'))

    def _handle_nan_recovery(self,epoch):
        if self.loss is not None and not torch.isfinite(self.loss).all(): raise RuntimeError('Nonfinite loss; stopped without automatic retry')
        return False

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.0)
        before=self.scaler.get_scale()
        self.scaler.step(self.optimizer);self.scaler.update();self.optimizer.zero_grad()
        if self.scaler.get_scale()>=before and self.ema:self.ema.update(self.model)

    def _build_train_pipeline(self):
        if getattr(self,'pipeline_created',False): raise RuntimeError('Refusing automatic OOM batch/optimizer rebuild')
        self.pipeline_created=True
        return super()._build_train_pipeline()

    def record_progress(self,trainer):
        write_json(self.save_dir/'progress.json',dict(epoch=self.epoch+1,total=self.epochs,updated_at=datetime.now().isoformat(),
                    losses=self.tloss.detach().cpu().tolist(),metrics=self.metrics))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true',help='Check split and fresh model on synthetic input only')
    p.add_argument('--resume',action='store_true',help='Resume this version weights/last.pt')
    a=p.parse_args();c=settings();torch.set_num_threads(4)
    if ',' in str(c['train']['device']):raise ValueError('This entry supports one GPU or CPU per version')
    if ultralytics.__version__!='8.4.98': raise RuntimeError('Use bundled Ultralytics 8.4.98')
    _,_,audit=validate_split(c);pre=(VERSION/c['pretrained']).resolve()
    if not pre.is_file(): raise FileNotFoundError(pre)
    if a.check:
        official=YOLO(str(pre)).model.float().cpu()
        m=build_model(str(CODE/'configs/yolo11m.yaml'),official,c['architecture']).eval()
        with torch.inference_mode(): result=m(torch.zeros(1,3,128,128))[0]
        assert torch.isfinite(result).all()
        write_json(CODE/'artifacts/model_check.json',dict(split=audit,architecture=c['architecture'],parameters=sum(p.numel() for p in m.parameters()),
                   finite_forward=True,training_started=False,official_pretrained_sha256=sha256(pre)))
        print('Split, class mapping, architecture and synthetic forward passed. No training started.')
        return
    data,prepared=check_ready(c,audit)
    contract=dict(configuration=c,split=audit['manifest_sha256'],preparation=prepared,
                  code=source_fingerprint(),official_pretrained=sha256(pre))
    record=CODE/'training_records/train';record.mkdir(parents=True,exist_ok=True)
    with run_lock(CODE/'artifacts/train.lock'):
        cp=record/'contract.json';last=VERSION/'weights/last.pt'
        if a.resume:
            if not last.exists() or not cp.exists(): raise FileNotFoundError('No complete checkpoint/contract to resume')
            if json.loads(cp.read_text(encoding='utf-8'))!=contract: raise ValueError('Training contract changed. Historical checkpoints retain the original contract; create a new version for training after the shared-path refactor. Do not overwrite the archived contract.')
        elif cp.exists() or any((VERSION/'weights').glob('*.pt')):
            raise FileExistsError('This version already has a run. Use --resume or a new version.')
        else: write_json(cp,contract)
        overrides=dict(c['train'],model=str(last if a.resume else CODE/'configs/yolo11m.yaml'),
                       pretrained=True if a.resume else str(pre),data=str(data),
                       project=str(record.parent),name=record.name,exist_ok=True)
        if a.resume: overrides['resume']=str(last)
        try:
            trainer=SplitTrainer(overrides=overrides,contract=contract,kind=c['architecture']);trainer.train()
        finally:
            from archive import refresh
            refresh()
if __name__=='__main__': main()
