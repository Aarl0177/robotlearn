# src/utils/env_utils.py
from __future__ import annotations

import importlib
from typing import Any, Dict

try:
    # Newer PettingZoo uses mpe2
    mpe2 = importlib.import_module("pettingzoo.mpe2")
    _use_mpe2 = True
except Exception:
    mpe2 = None
    _use_mpe2 = False

from typing import Callable

def make_simple_spread_env(N: int = 3, max_cycles: int = 200, local_ratio: float = 0.5,
                           continuous_actions: bool = True, render_mode: str | None = None):
    """Return a PettingZoo parallel_env for Simple Spread.
    Auto-handles old (mpe) vs new (mpe2) module name.
    """
    if _use_mpe2:
        simple_spread_v3 = importlib.import_module("pettingzoo.mpe2.simple_spread_v3")
    else:
        simple_spread_v3 = importlib.import_module("pettingzoo.mpe.simple_spread_v3")
    env = simple_spread_v3.parallel_env(
        N=N, max_cycles=max_cycles, local_ratio=local_ratio,
        continuous_actions=continuous_actions, render_mode=render_mode
    )
    return env
