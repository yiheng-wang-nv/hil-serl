"""Convenience entry points for Unitree-specific scripts."""

from .record_demos import main as record_demos_main  # noqa: F401
from .record_success_fail import main as record_success_fail_main  # noqa: F401
from .replay import main as replay_main  # noqa: F401

__all__ = [
    "record_demos_main",
    "record_success_fail_main",
    "replay_main",
]
