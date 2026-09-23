"""
tests/conftest.py

Makes `physai` importable directly from the repo's `src/` layout without
requiring `pip install -e .` first. Place this `tests/` directory as a
sibling of `src/` at the repo root (i.e. `<repo_root>/tests/conftest.py`,
`<repo_root>/src/physai/...`), or adjust SRC_DIR below to match your
layout.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Windows console encoding safety net.
#
# On Windows, Python's stdout/stderr default to the console's active
# codepage (often cp1252 or similar, not UTF-8) unless overridden. If
# *anything* printed during a test run -- an assertion message, a
# traceback, an error string from torch/jax/tensorflow/paddle or any
# other dependency -- contains a character outside that codepage (a
# smart quote, an arrow, a non-ASCII minus sign, etc.), the print itself
# raises UnicodeEncodeError ("'charmap' codec can't encode..."), which
# can abort the run independent of whether the underlying test passed.
#
# This isn't something we can fix by only cleaning our own source text
# (see test_pde_everything.py's docstrings/comments, which were already
# made pure ASCII) -- the offending text can come from anywhere in the
# dependency stack at runtime. The general fix is to make the streams
# themselves tolerant: reconfigure stdout/stderr to UTF-8 with a
# non-raising error handler, once, before pytest starts collecting or
# printing anything. `errors="backslashreplace"` means an unprintable
# character shows up as an escape sequence (e.g. \u2014) instead of
# crashing the write -- you lose nothing you need to debug a failure,
# you just never lose the whole run to an encoding error.
#
# `.reconfigure()` is a plain method on Python's standard text streams
# (available since 3.7) and also works on pytest's own capture-replaced
# stdout/stderr objects; wrapped in try/except in case some other tool
# in the plugin chain has already swapped in a stream that doesn't
# support it, so this can never itself become a new failure mode.
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")

if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Force CPU-only for torch/jax/tensorflow, set as early as physically
# possible in the pytest process.
#
# This must live in conftest.py, not just at the top of a single test
# module: pytest always loads conftest.py before collecting *any* test
# module in this directory, but test module collection order is not
# guaranteed to put any particular module first. If another test file
# (test_pde_residual.py, tests/test-trainer-backend/jax_train_test.py,
# etc.) gets collected first and imports torch/jax without forcing CPU,
# that import initializes the CUDA driver with the GPU visible -- and
# setting CUDA_VISIBLE_DEVICES afterward, from test_pde_everything.py's
# own module top, does nothing: once a process has initialized CUDA
# against a set of visible devices, env vars can't retroactively hide
# them from it. Every worker process the hard-geometry ProcessPoolExecutor
# then spawns inherits that same GPU-visible environment, and all of them
# fight over the same VRAM -- which is exactly what a long uniform run of
# `torch.AcceleratorError: CUDA error: out of memory` across unrelated
# PDEs looks like: not per-equation instability, just every worker
# process trying to claim GPU memory from the same card at once.
#
# `os.environ.setdefault` (not a hard overwrite) so a deliberate choice
# made before invoking pytest is respected rather than silently
# overridden.
# ---------------------------------------------------------------------------

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")           # torch, and most CUDA-based libs
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")            # jax
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")   # extra safety net for tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")            # quiet tensorflow's own startup logging