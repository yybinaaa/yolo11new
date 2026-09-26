from __future__ import annotations
import os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/"artifacts"/"ultralytics"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"artifacts"/"matplotlib"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE","1")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True,exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True,exist_ok=True)
