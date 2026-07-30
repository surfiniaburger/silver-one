"""Thread-limits environment setup.

Ensures OpenMP and numeric thread pools are constrained before importing numpy,
PyTorch, XGBoost, or SetFit to prevent thread pool initialization conflicts and
SIGSEGV (139) on macOS ARM64.
"""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
