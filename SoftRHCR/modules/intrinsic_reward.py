from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


def _safe_bool(aux: Optional[Mapping[str, Any]], key: str, default: bool = False) -> bool:
    if not isinstance(aux, Mapping):
        return bool(default)
    try:
        return bool(aux.get(key, default))
    except Exception:
        return bool(default)


def _safe_int(aux: Optional[Mapping[str, Any]], key: str, default: int = 0) -> int:
    if not isinstance(aux, Mapping):
        return int(default)
    try:
        return int(aux.get(key, default))
    except Exception:
        return int(default)


def _compute_aggressive_intrinsic_rewards(
    next_obs: Optional[Dict[str, Any]],
    prev_obs_aux: Optional[Dict[str, Any]] = None,
    next_obs_aux: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(next_obs, dict):
        return out
    for aid in next_obs.keys():
        prev_aux = prev_obs_aux.get(aid) if isinstance(prev_obs_aux, dict) else None
        next_aux = next_obs_aux.get(aid) if isinstance(next_obs_aux, dict) else None
        kpc_prev = _safe_int(prev_aux, "k_path_conflict", 0)
        kpc_now = _safe_int(next_aux, "k_path_conflict", 0)
        stagnant = _safe_bool(next_aux, "stagnant", False)

        reward = 0.0

        # Environment reward already covers step/task/progress/invalid/conflict terms.
        # Aggressive intrinsic reward only adds planner-aware shaping.
        reward += -0.30 * float(kpc_prev == 0 and kpc_now > 0)
        reward += -0.20 * float(kpc_prev > 0 and kpc_now > kpc_prev)

        reward += 0.25 * float(np.clip(kpc_prev - kpc_now, -1, 1))
        reward += 0.50 * float(kpc_prev > 0 and kpc_now == 0)
        reward += 0.20 * float(kpc_prev > 1 and kpc_now == 1)

        reward += -0.15 * float(kpc_prev > 0 and stagnant)
        out[str(aid)] = float(reward)
    return out


def compute_cr_intrinsic_rewards(
    next_obs: Optional[Dict[str, Any]],
    prev_obs_aux: Optional[Dict[str, Any]] = None,
    next_obs_aux: Optional[Dict[str, Any]] = None,
    reward_mode: str = "aggressive",
) -> Dict[str, float]:
    """Unified entry point for intrinsic rewards.

    Reward shaping relies solely on the change in ``k_path_conflict`` stored in
    ``obs_aux``; it does not need the raw obs / info / map dimensions.
    """
    if not isinstance(next_obs, dict):
        return {}
    mode = str(reward_mode or "aggressive").strip().lower()
    if mode == "aggressive":
        return _compute_aggressive_intrinsic_rewards(
            next_obs=next_obs,
            prev_obs_aux=prev_obs_aux,
            next_obs_aux=next_obs_aux,
        )
    return {}
