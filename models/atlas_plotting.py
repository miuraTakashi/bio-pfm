"""
Compatibility shim: model scripts import ``atlas_plotting`` from the ``python/`` directory.

Implementation lives in :mod:`lib.atlas_plotting`.
"""
from lib.atlas_plotting import *  # noqa: F403
from lib import atlas_plotting as _impl

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
