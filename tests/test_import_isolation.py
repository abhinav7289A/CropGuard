"""The serving image must never need torch.

`pip install ".[serve]"` pulls onnxruntime (~50MB) instead of torch (~2GB); that gap is what
keeps the container inside Render's free tier. A single stray `import torch` in
cropguard.serving would break the deployment — and it would break it *in production*, since
every dev machine has torch installed and nothing local would fail.

These tests import the modules in a subprocess with torch blocked, which is the only way to
catch it when torch is importable in the test environment.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Heavy training-only dependencies that must not reach the serving or evaluation packages.
FORBIDDEN = ("torch", "torchvision", "pytorch_lightning", "timm")

TORCH_FREE_MODULES = [
    "cropguard",
    "cropguard.config",
    "cropguard.serving.app",
    "cropguard.serving.model_loader",
    "cropguard.evaluation.hypothesis",
    "cropguard.evaluation.compare",
    "cropguard.evaluation.predict",
]

_BLOCKER = """
import sys

class Blocked(Exception):
    pass

class Blocker:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in {forbidden!r} else None
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {forbidden!r}:
            raise Blocked(
                "%s imported a forbidden dependency: %s" % ({module!r}, name)
            )
        return None

sys.meta_path.insert(0, Blocker())
import {module}
print("OK")
"""


@pytest.mark.parametrize("module", TORCH_FREE_MODULES)
def test_module_imports_without_torch(module: str):
    """Importing this module with torch blocked must still succeed."""
    code = _BLOCKER.format(forbidden=set(FORBIDDEN), module=module)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert proc.returncode == 0, (
        f"{module} cannot be imported without torch — this breaks the serving image.\n"
        f"{proc.stderr[-1500:]}"
    )
    assert "OK" in proc.stdout


def test_the_blocker_actually_blocks():
    """Guard the guard: if the import hook silently stopped working, every test above
    would pass vacuously."""
    code = _BLOCKER.format(forbidden=set(FORBIDDEN), module="cropguard.training.module")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    # cropguard.training.module imports torch at module level, so this MUST fail.
    assert proc.returncode != 0, "The import blocker is not working — the other tests are void"
