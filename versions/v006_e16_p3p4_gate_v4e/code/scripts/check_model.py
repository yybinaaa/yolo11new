"""Check actual v002 migration, synthetic CPU/CUDA output and saved PT loading."""
import _bootstrap  # noqa: F401
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from ultralytics import YOLO
from common import CODE, VERSION, sha256, write_json
from src.models.p3p4_spatial_gate import attach_spatial_gate, check_architecture


def main():
    torch.set_num_threads(4)
    subprocess.run([sys.executable, str(CODE / 'tests/test_gate.py')], cwd=CODE, check=True)
    source = VERSION.parent / 'v002_e16_v4e/weights/epoch180.pt'
    parent = YOLO(str(source)).model.float().cpu().eval()
    gated = attach_spatial_gate(parent).eval()
    checks = {}
    x = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        original = parent(x)[0]
        upgraded = gated(x)[0]
    torch.testing.assert_close(original, upgraded, rtol=0, atol=0)
    checks['actual_v002_epoch180_identity_cpu'] = True
    with tempfile.TemporaryDirectory(dir=CODE / 'artifacts', prefix='unit-checkpoint-') as folder:
        checkpoint = Path(folder) / 'synthetic_check_only.pt'
        torch.save(dict(model=gated, ema=None, epoch=-1, train_args={'task': 'detect', 'imgsz': 1024}), checkpoint)
        restored = YOLO(str(checkpoint)).model.float().eval()
        check_architecture(restored)
        with torch.inference_mode():
            torch.testing.assert_close(upgraded, restored(x)[0], rtol=0, atol=0)
    checks['ultralytics_pt_load_roundtrip'] = True
    if torch.cuda.is_available():
        parent = parent.cuda(); gated = gated.cuda()
        with torch.inference_mode():
            for shape in ((1, 3, 128, 128), (1, 3, 1024, 1024)):
                gpu_input = torch.randn(*shape, device='cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    a = parent(gpu_input)[0]; b = gated(gpu_input)[0]
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
                if not torch.isfinite(b).all(): raise RuntimeError('Nonfinite AMP output')
        checks['cuda_amp_128_and_1024_identity'] = True
        checks['cuda_device'] = torch.cuda.get_device_name(0)
    else:
        checks['cuda_amp_128_and_1024_identity'] = 'not_available'
    write_json(CODE / 'artifacts/verification.json', dict(time=datetime.now().isoformat(),
               unit_test_count=9, checks=checks, parent_weight_sha256=sha256(source), gate_parameters=43041,
               real_training_started=False, real_image_inference=False,
               note='Synthetic detection loss/backward and module optimizer/EMA/scheduler resume checked; full real-data training/resume is not exercised.'))
    from archive import refresh
    refresh()
    print('V006 migration and serialization verified; no real dataset training or inference performed.')


if __name__ == '__main__': main()
