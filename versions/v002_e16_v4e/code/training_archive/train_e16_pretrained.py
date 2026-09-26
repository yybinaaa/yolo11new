"""Independent E16 full training with atomic, resumable epoch checkpoints."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import _bootstrap  # noqa: F401
import psutil
import torch
import yaml
import ultralytics
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import unwrap_model
from src.models.multiscale_dw_output import TARGET, attach_multiscale_branch, check_architecture
from scripts.run_e15_multiscale_dw_continuous import (
    snapshot, cpu_tree, save_checkpoint, write_json,
    optimizer_group_names, sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/experiments/E16_pretrained_180.yaml'


def settings():
    c = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    paths = [Path(__file__), CONFIG, ROOT/c['pretrained'], ROOT/c['model'], ROOT/c['data'],
             ROOT/'src/models/multiscale_dw_branch.py', ROOT/'src/models/multiscale_dw_output.py', ROOT/'scripts/run_e15_multiscale_dw_continuous.py']
    contract = {'config': c, 'version': ultralytics.__version__,
                'files': {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}}
    contract['fingerprint'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return c, contract


class FreshTrainer(DetectionTrainer):
    def __init__(self, *args, contract, run, smoke_stop=None, **kwargs):
        self.contract, self.run, self.smoke_stop = contract, run, smoke_stop
        self.batches = 0
        super().__init__(*args, **kwargs)
        self.add_callback('on_train_start', self.initialize_scaler)
        self.add_callback('on_train_batch_end', self.progress)
        self.add_callback('on_train_epoch_end', self.smoke_end)

    def initialize_scaler(self, trainer):
        if not self.resume:
            self.scaler = torch.amp.GradScaler('cuda', enabled=bool(self.amp), init_scale=64)

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.scaler.get_scale() >= before:
            self.ema.update(self.model)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        model = attach_multiscale_branch(model)
        if weights is None:
            raise RuntimeError('Pretrained or resume weights are required')
        model.load(weights)
        check_architecture(model)
        if not self.resume:
            src, dst = weights.float().state_dict(), model.state_dict()
            matched = [k for k,v in src.items() if k in dst and v.shape == dst[k].shape]
            if not matched or not all(torch.equal(src[k], dst[k]) for k in matched):
                raise RuntimeError('Pretrained transfer verification failed')
            if model.get_submodule(TARGET).ms_branch.project.weight.count_nonzero():
                raise RuntimeError('New projection should start at zero')
            write_json(self.run/'initialization_report.json', {
                'source': str(ROOT/self.contract['config']['pretrained']),
                'matched_state_items': len(matched), 'target_state_items': len(dst),
                'source_classes': len(weights.names), 'target_classes': self.data['nc'],
                'new_branch_zero_projection': True, 'fresh_optimizer_and_ema': True})
        return model

    def check_resume(self, overrides):
        super().check_resume(overrides)
        self.args.project = str(self.run.parent)
        self.args.name = self.run.name
        self.args.save_dir = str(self.run)
        if self.resume:
            # Avoid replacing saved weights with the official pretrained file.
            self.args.pretrained = True

    def _load_checkpoint_state(self, ckpt):
        meta = ckpt.get('e16_fresh', {})
        if meta.get('fingerprint') != self.contract['fingerprint'] or meta.get('run') != self.run.name:
            raise RuntimeError('Checkpoint belongs to a different run or configuration')
        if optimizer_group_names(self.optimizer, self.model) != meta['group_names']:
            raise RuntimeError('Optimizer parameter order changed')
        super()._load_checkpoint_state(ckpt)
        unwrap_model(self.model).load_state_dict(ckpt['model'].float().state_dict(), strict=True)
        if self.scaler.state_dict() != ckpt['scaler']:
            raise RuntimeError('AMP state restore failed')
        write_json(self.run/'resume_report.json', {
            'next_epoch': ckpt['epoch']+2, 'target_epoch': self.args.epochs,
            'optimizer_states': len(self.optimizer.state), 'ema_updates': self.ema.updates,
            'live_model_restored': True, 'optimizer_and_scaler_restored': True})

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        self.metrics = {}
        if self.resume:
            self.scheduler.load_state_dict(ckpt['scheduler'])

    def _build_train_pipeline(self):
        if getattr(self, '_pipeline_built', False):
            raise RuntimeError('OOM: refusing to silently rebuild optimizer or change batch size')
        self._pipeline_built = True
        return super()._build_train_pipeline()

    def _handle_nan_recovery(self, epoch):
        if self.loss is not None and not torch.isfinite(self.loss).all():
            raise RuntimeError('Nonfinite loss; retained last complete checkpoint')
        return False

    def validate(self):
        return {}, None

    def final_eval(self):
        # Keep optimizer/scaler: the standard final_eval strips these states.
        return None

    def smoke_end(self, trainer):
        if self.smoke_stop and self.epoch+1 >= self.smoke_stop:
            self.stop = True

    def progress(self, trainer):
        self.batches += 1
        if self.batches == 1 or self.batches % 50 == 0:
            write_json(self.run/'progress.json', {'status':'training', 'pid':os.getpid(),
                'updated_at':datetime.now().isoformat(), 'epoch':self.epoch+1, 'target_epoch':self.args.epochs,
                'batches_this_process':self.batches, 'loss':self.tloss.detach().cpu().tolist(),
                'lrs':[g['lr'] for g in self.optimizer.param_groups], 'ema_updates':self.ema.updates})

    def save_model(self):
        if len(self.optimizer.state) != sum(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Incomplete AdamW states; no valid full update was saved')
        for values in self.optimizer.state.values():
            for value in values.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise RuntimeError('Nonfinite optimizer checkpoint')
        payload = {'epoch':self.epoch, 'model':snapshot(self.model), 'ema':snapshot(self.ema.ema),
            'updates':self.ema.updates, 'optimizer':cpu_tree(self.optimizer.state_dict()),
            'scaler':self.scaler.state_dict(), 'scheduler':deepcopy(self.scheduler.state_dict()),
            'train_args':vars(self.args), 'train_metrics':{}, 'best_fitness':None,
            'train_results':self.read_results_csv(), 'version':ultralytics.__version__,
            'date':datetime.now().isoformat(), 'e16_fresh':{
                'fingerprint':self.contract['fingerprint'], 'run':self.run.name,
                'group_names':optimizer_group_names(self.optimizer,self.model)}}
        save_checkpoint(self.last, payload)
        c = self.contract['config']
        epoch = self.epoch+1
        if epoch >= c['save_start'] and (epoch-c['save_start']) % c['save_interval'] == 0:
            target = self.wdir/f'epoch{epoch}.pt'
            if target.exists():
                raise FileExistsError(target)
            save_checkpoint(target, payload)
        write_json(self.run/'checkpoint_status.json', {'completed_epoch':epoch,
            'optimizer_states':len(self.optimizer.state), 'ema_updates':self.ema.updates,
            'last':str(self.last), 'updated_at':datetime.now().isoformat()})
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-stop', type=int, choices=(1,2))
    parser.add_argument('--resume', action='store_true', help='Require a saved last.pt')
    args = parser.parse_args()
    torch.set_num_threads(4)
    c, contract = settings()
    if ultralytics.__version__ != '8.4.98':
        raise RuntimeError('Unexpected Ultralytics version')
    run = ROOT/'runs/yolo11'/(c['run_name'] if not args.smoke_stop else 'smoke_e16_fresh_'+contract['fingerprint'][:8])
    run.mkdir(parents=True, exist_ok=True)
    contract_path = run/'run_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text(encoding='utf-8')) != contract:
        raise RuntimeError('Existing run uses different configuration/code')
    lock = run/'pipeline.lock'
    if lock.exists():
        old = json.loads(lock.read_text())
        try:
            proc = psutil.Process(old['pid'])
            active = abs(proc.create_time()-old['created']) < .01
        except psutil.NoSuchProcess:
            active = False
        if active:
            raise RuntimeError('This run is already active')
        lock.unlink()
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:
        json.dump({'pid':os.getpid(),'created':psutil.Process().create_time()}, stream)
    write_json(contract_path, contract)
    def status(stage, **extra):
        write_json(run/'pipeline_status.json', dict(status=stage, pid=os.getpid(),
                   updated_at=datetime.now().isoformat(), **extra))
    try:
        last = run/'weights/last.pt'
        if args.resume and not last.exists():
            raise FileNotFoundError('No checkpoint exists to resume')
        completed = -1
        if last.exists():
            ckpt = torch_load(last, map_location='cpu')
            completed = ckpt['epoch']
            del ckpt
        target = args.smoke_stop or c['epochs']
        if completed+1 < target:
            overrides = {k:v for k,v in c.items() if k not in {'run_name','save_start','save_interval'}}
            for key in ('model','pretrained','data'):
                overrides[key] = str(ROOT/overrides[key])
            overrides.update(project=str(run.parent),name=run.name,exist_ok=True,save=True,
                             save_period=-1,val=False,plots=False,patience=0,amp=True,compile=False)
            if args.smoke_stop:
                overrides.update(batch=1,workers=0,fraction=.005)
            if last.exists():
                overrides.update(resume=str(last),pretrained=True)
            status('training')
            trainer = FreshTrainer(overrides=overrides,contract=contract,run=run,smoke_stop=args.smoke_stop)
            trainer.train()
            del trainer
        saved = torch_load(last,map_location='cpu')
        if saved['epoch']+1 != target:
            raise RuntimeError('Training endpoint missing')
        del saved
        if args.smoke_stop:
            status('smoke_complete',completed_epoch=target)
            return
        write_json(run/'completion_report.json', {'completed_epoch':target,'status':'complete',
                   'finished_at':datetime.now().isoformat()})
        output = ROOT/f'outputs/yolo11/e16_official_pretrained_epoch{target}_v4e'
        summary_path = output/'inference_summary.json'
        if not summary_path.exists():
            status('inferencing')
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/predict_e15_epoch200_v4e.py'),
                '--weights',str(run/f'weights/epoch{target}.pt'),'--output',str(output)],cwd=ROOT,check=True)
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary['images'] != 669:
            raise RuntimeError('Test image count mismatch')
        status('complete',completed_epoch=target,output=str(output),images=summary['images'],detections=summary['detections'],submission_zip=str(output/'v4e_submission.zip'))
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
