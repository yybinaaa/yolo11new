"""V006 full-data training. Default action is check-only; --train starts training."""
import _bootstrap  # noqa: F401
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
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import unwrap_model

from common import CODE, VERSION, NAMES, settings, validate_training_data, sha256, write_json, source_fingerprint, run_lock
from src.models.p3p4_spatial_gate import attach_spatial_gate, check_architecture


def build_model(weights=None, verbose=False):
    model = DetectionModel(str(CODE / 'configs/yolo11m.yaml'), nc=9, ch=3, verbose=verbose)
    model = attach_spatial_gate(model)
    model.names = dict(enumerate(NAMES))
    if weights is not None:
        model.load(weights)
    check_architecture(model)
    return model


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def snapshot(model):
    result = deepcopy(unwrap_model(model)).float().cpu()
    if hasattr(result, 'criterion'): result.criterion = None
    if any(not torch.isfinite(v).all() for v in result.state_dict().values()):
        raise RuntimeError('Nonfinite checkpoint weights')
    return result


def atomic_save(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(payload, temp); os.replace(temp, path)


def optimizer_names(optimizer, model):
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


class GateTrainer(DetectionTrainer):
    def __init__(self, *args, contract, **kwargs):
        self.contract = contract
        super().__init__(*args, **kwargs)
        self.wdir = VERSION / 'weights'
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'
        nested = self.save_dir / 'weights'
        if nested.is_dir() and not any(nested.iterdir()): nested.rmdir()
        self.add_callback('on_train_start', self.initialize_runtime)
        self.add_callback('on_train_epoch_end', self.record_progress)

    def check_resume(self, overrides):
        super().check_resume(overrides)
        self.args.project = str(CODE / 'training_records')
        self.args.name = 'train'; self.args.save_dir = str(CODE / 'training_records/train')
        self.args.exist_ok = True
        if self.resume: self.args.pretrained = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None: raise RuntimeError('Initialization or resume weights required')
        model = build_model(weights, verbose)
        if not self.resume:
            src = weights.float().state_dict(); dst = model.state_dict()
            kind = self.contract['configuration']['initialization_kind']
            has_branch = any(k.startswith('model.16.ms_branch.') for k in src)
            if has_branch != (kind == 'e16') or any('spatial_gate' in k for k in src):
                raise RuntimeError('Initialization kind does not match source; use --resume for an existing V006 run')
            matched = [k for k, v in src.items() if k in dst and v.shape == dst[k].shape]
            if not matched or not all(torch.equal(src[k], dst[k]) for k in matched):
                raise RuntimeError('Existing parameter transfer failed')
            new_gate = model.model[16].spatial_gate.logits
            if new_gate.weight.count_nonzero() or new_gate.bias.count_nonzero():
                raise RuntimeError('Gate must start at identity')
            if kind == 'official' and model.model[16].ms_branch.project.weight.count_nonzero():
                raise RuntimeError('Fresh E16 projection must start at zero')
            if kind == 'e16' and any(k not in matched for k in src):
                raise RuntimeError('E16 transfer must preserve every state item')
            write_json(self.save_dir / 'initialization.json', dict(source_kind=kind, transferred_items=len(matched),
                       gate_identity=True, fresh_optimizer=True, gate_parameters=43041))
        return model

    def initialize_runtime(self, trainer):
        if not self.resume:
            self.scaler = torch.amp.GradScaler('cuda', enabled=bool(self.amp), init_scale=64)
        rng = getattr(self, 'pending_rng', None)
        if rng:
            random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
            if torch.cuda.is_available() and rng['cuda'] is not None: torch.cuda.set_rng_state_all(rng['cuda'])

    def _load_checkpoint_state(self, ckpt):
        meta = ckpt.get('v006', {})
        if meta.get('contract') != self.contract:
            raise RuntimeError('Checkpoint configuration, data or source hashes changed')
        if meta['optimizer_names'] != optimizer_names(self.optimizer, self.model):
            raise RuntimeError('Optimizer parameter ordering changed')
        super()._load_checkpoint_state(ckpt)
        unwrap_model(self.model).load_state_dict(ckpt['model'].float().state_dict(), strict=True)
        self.pending_rng = ckpt['rng']
        write_json(self.save_dir / 'resume.json', dict(completed_epoch=ckpt['epoch'] + 1,
                   restored=['live_model', 'optimizer', 'ema', 'scaler'],
                   note='Epoch-boundary resume; worker augmentation order need not be bitwise identical.'))

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        if self.resume:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.metrics = {}

    def validate(self):
        # Upstream can request validation on the last epoch even when val=False.
        return {}, None

    def final_eval(self):
        write_json(self.save_dir / 'completion.json', dict(completed_epoch=self.epoch + 1,
                   status='complete' if self.epoch + 1 >= self.epochs else 'stopped_early',
                   evaluated=False, note='No local validation, no best.pt selection and no automatic V4E inference.'))

    def save_model(self):
        for values in self.optimizer.state.values():
            for value in values.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise RuntimeError('Nonfinite optimizer state')
        payload = dict(epoch=self.epoch, model=snapshot(self.model), ema=snapshot(self.ema.ema),
                       updates=self.ema.updates, optimizer=cpu_tree(self.optimizer.state_dict()),
                       scaler=self.scaler.state_dict(), scheduler=deepcopy(self.scheduler.state_dict()),
                       train_args=vars(self.args), train_metrics={}, best_fitness=None,
                       train_results=self.read_results_csv(), version=ultralytics.__version__, date=datetime.now().isoformat(),
                       rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                       v006=dict(contract=self.contract, optimizer_names=optimizer_names(self.optimizer, self.model)))
        atomic_save(self.last, payload)
        epoch = self.epoch + 1
        if epoch in self.contract['configuration']['save_epochs'] or epoch == self.epochs:
            target = self.wdir / f'epoch{epoch}.pt'
            if target.exists(): raise FileExistsError(target)
            atomic_save(target, payload)
        write_json(self.save_dir / 'checkpoint_status.json', dict(completed_epoch=epoch, last=str(self.last)))
        return True

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        before = self.scaler.get_scale()
        self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad()
        if self.scaler.get_scale() >= before and self.ema: self.ema.update(self.model)

    def _handle_nan_recovery(self, epoch):
        if self.loss is not None and not torch.isfinite(self.loss).all():
            raise RuntimeError('Nonfinite loss; stopped without automatic retry')
        return False

    def _build_train_pipeline(self):
        if getattr(self, 'pipeline_created', False): raise RuntimeError('Refusing automatic OOM optimizer rebuild')
        self.pipeline_created = True
        return super()._build_train_pipeline()

    def record_progress(self, trainer):
        write_json(self.save_dir / 'progress.json', dict(epoch=self.epoch + 1, total=self.epochs,
                   updated_at=datetime.now().isoformat(), losses=self.tloss.detach().cpu().tolist(), evaluated=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args(); c = settings(); torch.set_num_threads(4)
    if ultralytics.__version__ != '8.4.98': raise RuntimeError('Use version-local Ultralytics 8.4.98')
    if ',' in str(c['train']['device']): raise ValueError('One GPU or CPU per run is supported')
    pre = (VERSION / c['pretrained']).resolve()
    if not pre.is_file(): raise FileNotFoundError(pre)
    data, audit = validate_training_data(c)
    contract = dict(configuration=c, data=audit, source=source_fingerprint(), initialization_sha256=sha256(pre))
    if not args.train and not args.resume:
        source = YOLO(str(pre)).model.float().cpu()
        # Exercise the same get_model implementation as actual training.
        trainer = object.__new__(GateTrainer)
        trainer.resume = False; trainer.contract = contract; trainer.save_dir = CODE / 'artifacts/check'
        model = trainer.get_model(weights=source, verbose=False).eval()
        with torch.inference_mode(): prediction = model(torch.zeros(1, 3, 128, 128))[0]
        if not torch.isfinite(prediction).all(): raise RuntimeError('Nonfinite synthetic prediction')
        write_json(CODE / 'artifacts/model_check.json', dict(architecture=c['architecture'], data=audit,
                   parameters=sum(p.numel() for p in model.parameters()), gate_parameters=43041,
                   initialization_sha256=sha256(pre), synthetic_forward=True, training_started=False))
        print('V006 data, trainer construction and synthetic forward checks passed. No training started.')
        from archive import refresh
        refresh()
        return
    record = CODE / 'training_records/train'
    with run_lock(CODE / 'artifacts/train.lock'):
        contract_path = record / 'contract.json'; last = VERSION / 'weights/last.pt'
        if args.resume:
            if not last.exists() or not contract_path.exists(): raise FileNotFoundError('No V006 run to resume')
            if json.loads(contract_path.read_text(encoding='utf-8')) != contract: raise RuntimeError('Training contract changed')
            ckpt = torch_load(last, map_location='cpu')
            if ckpt['epoch'] + 1 >= c['train']['epochs']: raise RuntimeError('Run already completed')
            del ckpt
        elif contract_path.exists() or any((VERSION / 'weights').glob('*.pt')):
            raise FileExistsError('Run already exists; use --resume or create another version')
        else:
            write_json(contract_path, contract)
        overrides = dict(c['train'], model=str(last if args.resume else CODE / 'configs/yolo11m.yaml'),
                         pretrained=True if args.resume else str(pre), data=str(data),
                         project=str(record.parent), name='train', exist_ok=True)
        if args.resume: overrides['resume'] = str(last)
        try:
            trainer = GateTrainer(overrides=overrides, contract=contract)
            trainer.train()
        except BaseException as exc:
            write_json(record / 'failure.json', dict(error=repr(exc), time=datetime.now().isoformat()))
            raise
        finally:
            from archive import refresh
            refresh()


if __name__ == '__main__':
    main()
