"""Runtime bootstrap inherited by Windows DataLoader worker processes."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def _preload_msvcp() -> ctypes.CDLL | None:
    override = os.environ.get("STEEL_VC_RUNTIME_DIR")
    candidates = []
    if override:
        candidates.append(Path(override) / "msvcp140.dll")
    candidates.extend(
        [
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "native"
            / component
            / subdirectory
            / "bin"
            / "msvcp140.dll"
            for component, subdirectory in (
                ("poppler", "Library"),
                ("libheif", "libheif"),
                ("jxrlib", "jxrlib"),
            )
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return ctypes.CDLL(str(candidate))
    return None


# The module-level reference keeps the DLL loaded until the worker exits.
_MSVCP_RUNTIME_HANDLE = _preload_msvcp()
