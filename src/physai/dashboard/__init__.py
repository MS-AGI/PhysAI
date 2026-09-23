"""
physai/dashboard

Optional live terminal dashboard for training runs. Nothing in
``physai.dashboard`` is imported by ``physai`` core, so ``rich`` and
``llama-cpp-python`` stay optional dependencies — installed only if you
use this subpackage.
"""
from .live import RichDashboardCallback

__all__ = ["RichDashboardCallback"]