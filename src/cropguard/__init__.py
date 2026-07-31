"""CropGuard — production MLOps pipeline for crop disease detection.

Subpackages are import-isolated by environment:
- cropguard.data      -> pip install .[data]
- cropguard.training  -> pip install .[train]
- cropguard.serving   -> pip install .[serve]

Keep this module free of heavy imports so the serving image never needs torch.
"""

__version__ = "0.1.0"
