import math
from typing import Callable, Dict, Tuple, Any, Optional

import torch


def _get_single_or_nested(data: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
    return data if data is not None else default


def apply_soft_rhcr_defaults(env_cfg: Any, algo_cfg: Any) -> None:
    try:
        env_params = getattr(env_cfg, "train", None) if env_cfg is not None else None
        if env_params is None:
            return
        n_agents = int(getattr(env_params, "num_agvs", 0))
        map_size_str = str(getattr(env_params, "map_size", "long") or "long").strip().lower()
        if map_size_str == "long":
            max_x, max_y = 10, 5
        elif map_size_str == "wide":
            max_x, max_y = 7, 7
        elif map_size_str == "compact":
            max_x, max_y = 5, 5
        else:
            max_x, max_y = 7, 7

        k_horizon = int(getattr(algo_cfg, "soft_rhcr_k", 10))
        if k_horizon <= 0:
            return

        planner_coverage = int(getattr(algo_cfg, "soft_rhcr_L", 2))
        if planner_coverage <= 0:
            planner_coverage = 2

        planner_horizon = int(n_agents * max(max_x, max_y) * planner_coverage)
        planner_horizon = max(planner_horizon, k_horizon * 2)
        try:
            planner_args = getattr(env_cfg.planner, "overrides", {})
        except Exception:
            planner_args = {}
        if planner_args is None:
            planner_args = {}
        existing_horizon = planner_args.get("horizon")
        should_fill_horizon = existing_horizon is None
        if not should_fill_horizon:
            try:
                should_fill_horizon = int(existing_horizon) <= 0
            except Exception:
                should_fill_horizon = False
        if should_fill_horizon:
            planner_args["horizon"] = planner_horizon
        try:
            env_cfg.planner.overrides = planner_args
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SoftRHCR auxiliary loss factory
# ---------------------------------------------------------------------------
def make_soft_rhcr_aux_loss(agent) -> Callable:
    """Create the SoftRHCR auxiliary loss function.

    The choice is driven by ``agent.planner_aux_loss``:
    - "consistency": fp consistency constraint
    - "kl":          legacy KL/BC imitation loss (reserved)
    - "none":        no auxiliary loss

    The returned function has the signature::

        fn(data, flat_idx, new_logprobs) -> (loss_tensor, metrics_dict)
    """
    aux_mode = str(getattr(agent, "planner_aux_loss", "consistency")).strip().lower()

    def aux_loss(data, flat_idx: torch.Tensor, new_logprobs: torch.Tensor):
        if aux_mode == "none":
            return new_logprobs.sum() * 0.0, {}
        if aux_mode == "consistency":
            return _consistency_aux_loss(agent, data, flat_idx, new_logprobs)
        if aux_mode == "kl":
            return _kl_aux_loss(agent, data, flat_idx, new_logprobs)
        return new_logprobs.sum() * 0.0, {}

    return aux_loss


def _consistency_aux_loss(agent, data, flat_idx, new_logprobs):
    """FP consistency constraint loss."""
    coef = float(agent._current_fp_consistency_coef())
    if coef <= 0.0:
        return torch.zeros((), device=agent.device), {}

    use_fp = data.get("use_fp")
    if use_fp is None:
        return torch.zeros((), device=agent.device), {}

    idx_long = flat_idx.to(device=agent.device, dtype=torch.long)
    use_fp_mask = use_fp.reshape(-1).index_select(0, idx_long) > 0.5

    if bool(agent.fp_consistency_safe_only):
        kpc = data.get("k_path_conflict")
        if kpc is not None:
            kpc_mask = kpc.reshape(-1).index_select(0, idx_long) <= 0.5
            use_fp_mask = use_fp_mask & kpc_mask

    if not bool(use_fp_mask.any().item()):
        return torch.zeros((), device=agent.device), {}

    logp_fp = new_logprobs[use_fp_mask]
    p_min = max(1e-6, min(1.0, float(agent.fp_consistency_pmin)))
    log_p_min = torch.as_tensor(math.log(p_min), device=agent.device, dtype=logp_fp.dtype)
    consistency_loss = torch.relu(log_p_min - logp_fp).mean()

    return coef * consistency_loss, {
        "loss/fp_consistency": float(consistency_loss.item()),
        "loss/fp_consistency_coef": float(coef),
    }


def _kl_aux_loss(agent, data, flat_idx, new_logprobs):
    """Legacy KL/BC imitation loss (reserved)."""
    coef = float(agent._current_kl_coef())
    if coef <= 0.0:
        return torch.zeros((), device=agent.device), {}
    # Reserved: would need the planner action from data to compute KL / cross-entropy
    return torch.zeros((), device=agent.device), {}
