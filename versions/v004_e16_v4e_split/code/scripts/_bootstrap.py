"""Resolve this version's source and the Windows runtime before importing torch."""
import os
import sys
from pathlib import Path
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'scripts/runtime_bootstrap'))
import sitecustomize  # noqa: E402,F401
sys.path.insert(0, str(CODE))
os.environ['YOLO_CONFIG_DIR'] = str(CODE / 'artifacts/ultralytics')
os.environ['MPLCONFIGDIR'] = str(CODE / 'artifacts/matplotlib')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONPATH'] = os.pathsep.join([str(CODE/'scripts/runtime_bootstrap'),str(CODE)])
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
