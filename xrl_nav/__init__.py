"""Explainable RL for autonomous navigation."""
import os as _os

__version__ = "1.0.0"

# Small MLPs run fastest single-threaded; multi-threaded BLAS causes heavy contention on small batches.
try:
    import torch as _torch

    _torch.set_num_threads(int(_os.environ.get("XRL_TORCH_THREADS", "1")))
except ImportError:  # torch-free use of the environment
    pass
