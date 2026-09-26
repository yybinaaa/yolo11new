"""Synthetic unit checks only: no real-image inference or dataset training."""
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import _bootstrap  # noqa: F401
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import ModelEMA
from common import CODE, VERSION, settings
from src.models.multiscale_dw_output import attach_multiscale_branch
from src.models.p3p4_spatial_gate import P3P4SpatialGate, attach_spatial_gate, check_architecture
from train import GateTrainer, build_model


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(20260926)
        cls.base = DetectionModel(str(CODE / 'configs/yolo11m.yaml'), nc=9, verbose=False).eval()
        cls.e16 = attach_multiscale_branch(cls.base).eval()
        # Nonzero E16 residual is essential: zero residual would hide a wrong gate.
        nn.init.normal_(cls.e16.model[16].ms_branch.project.weight, std=.02)

    def test_gate_shape_identity_and_parameters(self):
        gate = P3P4SpatialGate(256, 512)
        x, s = torch.randn(2, 256, 13, 17), torch.randn(2, 512, 7, 9)
        g = gate(x, s)
        self.assertEqual(g.shape, (2, 1, 13, 17))
        self.assertTrue(torch.equal(g, torch.ones_like(g)))
        self.assertEqual(sum(p.numel() for p in gate.parameters()), 43041)

    def test_nonzero_e16_identity_and_source_unchanged(self):
        original_from = self.e16.model[16].f
        gated = attach_spatial_gate(self.e16).eval()
        old = self.e16.state_dict(); new = gated.state_dict()
        self.assertTrue(all(torch.equal(value, new[key]) for key, value in old.items()))
        self.assertTrue(all('spatial_gate' in key for key in new.keys() - old.keys()))
        self.assertEqual(self.e16.model[16].f, original_from)
        for shape in ((1, 3, 128, 128), (1, 3, 96, 160)):
            x = torch.randn(*shape)
            with torch.inference_mode():
                torch.testing.assert_close(self.e16(x)[0], gated(x)[0], rtol=0, atol=0)
        check_architecture(gated)

    def test_forward_split_and_fusion(self):
        gated = attach_spatial_gate(self.e16).eval()
        layer = gated.model[16]
        inputs = [torch.randn(1, layer.cv1.conv.in_channels, 12, 16), torch.randn(1, 512, 6, 8)]
        with torch.inference_mode():
            torch.testing.assert_close(layer(inputs), layer.forward_split(inputs), rtol=0, atol=0)
            x = torch.randn(1, 3, 128, 128)
            expected = gated(x)[0]
            gated.fuse(verbose=False)
            torch.testing.assert_close(expected, gated(x)[0], rtol=2e-4, atol=2e-4)
        check_architecture(gated)

    def test_gradient_reaches_context_and_all_gate_layers(self):
        layer = attach_spatial_gate(self.e16).model[16].train()
        optimizer = torch.optim.SGD(layer.parameters(), lr=.02)
        names = ['spatial_gate.logits.weight', 'spatial_gate.p3_proj.0.weight',
                 'spatial_gate.p4_proj.0.weight', 'spatial_gate.fuse.0.weight', 'ms_branch.project.weight']
        reached = set()
        for _ in range(4):
            inputs = [torch.randn(2, layer.cv1.conv.in_channels, 8, 8), torch.randn(2, 512, 4, 4)]
            optimizer.zero_grad()
            output = layer(inputs)
            loss = (output - torch.randn_like(output)).square().mean()
            loss.backward()
            for name, p in layer.named_parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    if p.grad.abs().sum() > 0: reached.add(name)
            optimizer.step()
        self.assertTrue(set(names).issubset(reached), set(names) - reached)

    def test_official_initialization_and_trainer_build(self):
        source = YOLO(str(VERSION / 'weights/initialization/yolo11m.pt')).model.float().cpu()
        trainer = object.__new__(GateTrainer)
        trainer.resume = False
        trainer.contract = {'configuration': settings()}
        with tempfile.TemporaryDirectory(dir=CODE / 'artifacts', prefix='unit-trainer-') as folder:
            trainer.save_dir = Path(folder)
            model = trainer.get_model(weights=source, verbose=False)
        check_architecture(model)
        self.assertEqual(model.model[16].ms_branch.project.weight.count_nonzero().item(), 0)
        self.assertFalse(settings()['train']['val'])
        self.assertEqual(trainer.validate(), ({}, None))

    def test_full_model_checkpoint_roundtrip_and_reconstruction(self):
        gated = attach_spatial_gate(self.e16).eval()
        with torch.no_grad():
            gated.model[16].spatial_gate.logits.weight.normal_(std=.01)
        data = io.BytesIO()
        torch.save({'model': gated}, data)
        data.seek(0)
        loaded = torch.load(data, map_location='cpu', weights_only=False)['model']
        rebuilt = build_model(loaded).eval()
        check_architecture(loaded); check_architecture(rebuilt)
        x = torch.randn(1, 3, 128, 128)
        with torch.inference_mode():
            torch.testing.assert_close(gated(x)[0], loaded(x)[0], rtol=0, atol=0)
            torch.testing.assert_close(gated(x)[0], rebuilt(x)[0], rtol=0, atol=0)

    def test_reject_double_install_and_wrong_architecture(self):
        with self.assertRaises(RuntimeError): check_architecture(self.e16)
        with self.assertRaises(TypeError): attach_spatial_gate(attach_spatial_gate(self.e16))

    def test_detection_loss_backward_on_synthetic_batch(self):
        model = attach_spatial_gate(self.e16).train()
        model.args = get_cfg()
        batch = dict(img=torch.rand(2, 3, 128, 128), batch_idx=torch.tensor([0., 1.]),
                     cls=torch.tensor([[0.], [8.]]), bboxes=torch.tensor([[.5, .5, .2, .2], [.4, .4, .1, .3]]))
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        grad = model.model[16].spatial_gate.logits.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_checkpoint_live_optimizer_ema_scheduler_restore(self):
        with tempfile.TemporaryDirectory(dir=CODE / 'artifacts', prefix='unit-resume-') as folder:
            def trainer_fixture():
                trainer = object.__new__(GateTrainer)
                trainer.model = attach_spatial_gate(self.e16).model[16].train()
                trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.001)
                trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1, gamma=.9)
                trainer.scaler = torch.amp.GradScaler('cuda', enabled=False)
                trainer.ema = ModelEMA(trainer.model)
                trainer.contract = {'configuration': {'save_epochs': []}}
                trainer.save_dir = Path(folder); trainer.wdir = Path(folder)
                trainer.last = Path(folder) / 'last.pt'; trainer.csv = Path(folder) / 'results.csv'
                trainer.args = SimpleNamespace(model=str(trainer.last), close_mosaic=0, epochs=3)
                trainer.epoch = 0; trainer.epochs = 3; trainer.resume = True
                return trainer
            first = trainer_fixture()
            layer = first.model
            batch = [torch.randn(2, layer.cv1.conv.in_channels, 8, 8), torch.randn(2, 512, 4, 4)]
            layer(batch).square().mean().backward()
            first.optimizer.step(); first.scheduler.step(); first.ema.update(layer)
            first.save_model()
            checkpoint = torch.load(first.last, map_location='cpu', weights_only=False)
            second = trainer_fixture()
            second.resume_training(checkpoint)
            self.assertEqual(second.start_epoch, 1)
            self.assertEqual(first.scheduler.state_dict(), second.scheduler.state_dict())
            self.assertEqual(first.ema.updates, second.ema.updates)
            for name, value in first.model.state_dict().items():
                torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)
            for name, value in first.ema.ema.state_dict().items():
                torch.testing.assert_close(value, second.ema.ema.state_dict()[name], rtol=0, atol=0)
            state1 = first.optimizer.state_dict(); state2 = second.optimizer.state_dict()
            self.assertEqual(state1['param_groups'], state2['param_groups'])
            for key, values in state1['state'].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, state2['state'][key][name], rtol=0, atol=0)
            second.contract = {'configuration': {'save_epochs': [99]}}
            with self.assertRaises(RuntimeError): second._load_checkpoint_state(checkpoint)


if __name__ == '__main__': unittest.main(verbosity=2)
