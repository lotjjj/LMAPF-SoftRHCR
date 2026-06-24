from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def _flatten_obs(obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
        for key, value in obs.items()
    }


def _flatten_2d(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.dim() < 2:
        return value.reshape(-1)
    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])


def _slice_obs(obs: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {key: value[idx] for key, value in obs.items()}


def _compute_policy_loss(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    clip_param: float,
):
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss


def _call_aux_loss(
    aux_loss_fn: Optional[Callable],
    data: Dict[str, torch.Tensor],
    flat_idx: torch.Tensor,
    new_logprobs: torch.Tensor,
):
    if aux_loss_fn is None:
        return new_logprobs.sum() * 0.0, {}
    return aux_loss_fn(data, flat_idx, new_logprobs)


def _compute_policy_update_mask(
    policy_mode: str,
    use_fp_flat: Optional[torch.Tensor],
    kpc_flat: Optional[torch.Tensor],
    idx: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Compute the policy-loss sample mask according to ``policy_update_mode``."""
    if policy_mode == "on_policy":
        if use_fp_flat is not None:
            return use_fp_flat[idx] <= 0.5
    elif policy_mode == "kpc_nonzero":
        if kpc_flat is not None:
            return kpc_flat[idx] > 0
    return None


def _compute_masked_policy_loss(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    clip_param: float,
    policy_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute the masked policy loss.

    When ``policy_mask`` is not None but is entirely False (e.g. in on_policy
    mode a mini-batch may consist solely of FP samples), return a zero-gradient
    loss to avoid computing policy gradients over FP samples.
    """
    if policy_mask is not None:
        if policy_mask.any():
            return _compute_policy_loss(ratios[policy_mask], advantages[policy_mask], clip_param)
        # All mask entries are False: return a zero loss while keeping the graph intact
        return (ratios * advantages * 0.0).mean()
    return _compute_policy_loss(ratios, advantages, clip_param)


def _compute_value_loss_mappo(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    agent,
) -> torch.Tensor:
    """Compute the MAPPO value loss (clipped or unclipped Huber loss)."""
    if bool(getattr(agent, "use_clipped_value_loss", True)):
        clipped_values = old_values + torch.clamp(
            new_values - old_values,
            -float(agent.clip_param),
            float(agent.clip_param),
        )
        value_loss_unclipped = F.huber_loss(
            new_values, returns.detach(),
            delta=float(agent.huber_delta), reduction="none",
        )
        value_loss_clipped = F.huber_loss(
            clipped_values, returns.detach(),
            delta=float(agent.huber_delta), reduction="none",
        )
        return 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
    return 0.5 * F.huber_loss(
        new_values, returns.detach(),
        delta=float(agent.huber_delta),
    )


def _freeze_batchnorm_running_stats(module: Optional[torch.nn.Module]) -> None:
    if module is None:
        return
    for submodule in module.modules():
        if isinstance(
            submodule,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        ):
            submodule.eval()


# ---------------------------------------------------------------------------
# Logger / gradient-monitoring helpers
# ---------------------------------------------------------------------------
def _get_logger(agent):
    """Retrieve the TrainLogger attached to *agent* (if any)."""
    return getattr(agent, "logger", None)


def _compute_grad_monitoring(
    agent,
    ppo_loss: torch.Tensor,
    aux_loss: torch.Tensor,
) -> Dict[str, float]:
    """Compute gradient monitoring metrics for three parameter groups.

    Groups
    ------
    - ``actor_extractor``:  extractor.parameters()  (shared encoder)
    - ``actor_head``:       actor.parameters()      (policy head)
    - ``actor_total``:      extractor + actor       (combined)

    For each group four metrics are produced:
    ``ppo_norm``, ``aux_norm``, ``aux_to_ppo_ratio``, ``aux_ppo_cos``.
    """
    metrics: Dict[str, float] = {}

    groups: Dict[str, list] = {
        "actor_extractor": list(agent.extractor.parameters()),
        "actor_head": list(agent.actor.parameters()),
        "actor_total": list(agent.extractor.parameters()) + list(agent.actor.parameters()),
    }

    # --- PPO gradients ---
    agent.optimizer.zero_grad()
    ppo_loss.backward(retain_graph=True)
    ppo_flat_grads: Dict[str, Optional[torch.Tensor]] = {}
    for name, params in groups.items():
        grads = [p.grad.detach().clone() for p in params if p.grad is not None]
        if grads:
            flat = torch.cat([g.flatten() for g in grads])
            ppo_flat_grads[name] = flat
            metrics[f"grad/{name}_ppo_norm"] = float(flat.norm().item())
        else:
            ppo_flat_grads[name] = None
            metrics[f"grad/{name}_ppo_norm"] = 0.0

    # --- Auxiliary gradients ---
    agent.optimizer.zero_grad()
    aux_loss.backward(retain_graph=True)
    for name, params in groups.items():
        grads = [p.grad.detach().clone() for p in params if p.grad is not None]
        if grads:
            flat = torch.cat([g.flatten() for g in grads])
            aux_norm = float(flat.norm().item())
            metrics[f"grad/{name}_aux_norm"] = aux_norm
            ppo_norm = metrics.get(f"grad/{name}_ppo_norm", 0.0)
            metrics[f"grad/{name}_aux_to_ppo_ratio"] = (
                aux_norm / ppo_norm if ppo_norm > 1e-10 else 0.0
            )
            ppo_flat = ppo_flat_grads.get(name)
            if ppo_flat is not None and ppo_flat.numel() == flat.numel():
                cos = float(
                    F.cosine_similarity(ppo_flat.unsqueeze(0), flat.unsqueeze(0)).item()
                )
                metrics[f"grad/{name}_aux_ppo_cos"] = cos
            else:
                metrics[f"grad/{name}_aux_ppo_cos"] = 0.0
        else:
            metrics[f"grad/{name}_aux_norm"] = 0.0
            metrics[f"grad/{name}_aux_to_ppo_ratio"] = 0.0
            metrics[f"grad/{name}_aux_ppo_cos"] = 0.0

    # Clean up — zero all gradients before normal training backward
    agent.optimizer.zero_grad()
    return metrics


def _finalize_metrics(
    total_loss: float,
    total_ppo_loss: float,
    total_policy_loss: float,
    total_value_loss: float,
    total_entropy_loss: float,
    total_aux_loss: float,
    total_approx_kl: float,
    total_clipfrac: float,
    total_grad_norm: float,
    aux_metrics_accum: Dict[str, float],
    n_batches: int,
    n_optimizer_steps: int,
    current_lr: float,
    batch_size: int,
    grad_monitor_metrics: Dict[str, float],
    grad_monitor_done: bool,
    logger,
) -> Dict[str, float]:
    """Assemble the final metrics dict from per-batch accumulators."""
    inv_b = 1.0 / max(1, n_batches)
    inv_opt = 1.0 / max(1, n_optimizer_steps)

    metrics: Dict[str, float] = {
        "loss": total_loss * inv_b,
        "loss/ppo": total_ppo_loss * inv_b,
        "loss/policy": total_policy_loss * inv_b,
        "loss/value": total_value_loss * inv_b,
        "loss/entropy": total_entropy_loss * inv_b,
        "ppo/approx_kl": total_approx_kl * inv_b,
        "ppo/clipfrac": total_clipfrac * inv_b,
        "optim/grad_norm": total_grad_norm * inv_opt,
        "optim/lr": float(current_lr),
        "optim/batch_size": float(batch_size),
    }

    # Auxiliary loss metrics (loss/fp_consistency, loss/kl, etc.)
    for k, v in aux_metrics_accum.items():
        metrics[k] = v * inv_b

    # Gradient monitoring metrics
    if grad_monitor_done:
        metrics.update(grad_monitor_metrics)
        metrics["grad/monitor_interval"] = float(
            logger.grad_monitor_interval if logger is not None else 5
        )

    return metrics


# ---------------------------------------------------------------------------
# IPPO update
# ---------------------------------------------------------------------------
def ippo_update(agent, aux_loss_fn=None) -> Dict[str, float]:
    batch = agent.buffer.get_all()
    if int(agent.buffer.ptr) <= 0:
        return {}

    logger = _get_logger(agent)
    do_grad_monitor = logger is not None and logger.should_monitor_grads()
    grad_monitor_metrics: Dict[str, float] = {}
    grad_monitor_done = False

    grad_clip_max_norm = getattr(agent, "grad_clip_max_norm", None)
    obs = _flatten_obs(batch["obs"])
    actions = batch["actions"].reshape(-1)
    old_logprobs = batch["logprobs"].reshape(-1)
    returns = agent.compute_returns(batch["rewards"], batch["values"], batch["dones"], batch["next_values"])
    advantages = (returns - batch["values"]).reshape(-1)
    returns = returns.reshape(-1)
    action_masks = _flatten_2d(batch.get("action_masks"))
    use_fp_flat = _flatten_2d(batch.get("use_fp"))
    kpc_flat = _flatten_2d(batch.get("k_path_conflict"))
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.detach()
    n_total = int(actions.shape[0])
    batch_size = int(getattr(agent, "batch_size", 0) or 0)
    resolved_batch_size = n_total if batch_size <= 0 else min(batch_size, n_total)

    agent.extractor.train()
    agent.actor.train()
    agent.critic.train()
    _freeze_batchnorm_running_stats(agent.extractor)
    _freeze_batchnorm_running_stats(agent.actor)
    _freeze_batchnorm_running_stats(agent.critic)

    # Metric accumulators
    total_loss = 0.0
    total_ppo_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_aux_loss = 0.0
    total_approx_kl = 0.0
    total_clipfrac = 0.0
    total_grad_norm = 0.0
    aux_metrics_accum: Dict[str, float] = {}
    n_batches = 0
    n_optimizer_steps = 0

    for _ in range(int(agent.ppo_epoch)):
        perm = torch.randperm(n_total, device=actions.device)
        for batch_start in range(0, n_total, resolved_batch_size):
            batch_end = min(batch_start + resolved_batch_size, n_total)
            idx = perm[batch_start:batch_end]
            agent.optimizer.zero_grad()

            b_obs = _slice_obs(obs, idx)
            b_actions = actions[idx]
            b_old_logprobs = old_logprobs[idx]
            b_advantages = advantages[idx]
            b_returns = returns[idx]
            b_action_masks = action_masks[idx] if action_masks is not None else None

            features = agent.extractor(b_obs)
            logits = agent.actor(features)
            if b_action_masks is not None:
                logits = logits.masked_fill(b_action_masks <= 0, -1e9)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(b_actions)
            entropies = dist.entropy()
            ratios = torch.exp(new_logprobs - b_old_logprobs)
            # --- Policy loss with sample filtering ---
            _policy_mode = getattr(agent, "policy_update_mode", "all")
            _policy_mask = _compute_policy_update_mask(_policy_mode, use_fp_flat, kpc_flat, idx)
            policy_loss = _compute_masked_policy_loss(
                ratios, b_advantages, float(agent.clip_param), _policy_mask
            )

            # Entropy: compute over all samples
            entropy = entropies.mean()

            new_values = agent.critic(features).squeeze(-1)
            value_loss = F.mse_loss(new_values, b_returns.detach())
            aux_loss, aux_metrics = _call_aux_loss(aux_loss_fn, batch, idx, new_logprobs)
            ppo_loss = (
                policy_loss
                + float(agent.value_loss_coef) * value_loss
                - float(agent.entropy_coef) * entropy
            )
            loss = ppo_loss + aux_loss

            # Gradient monitoring (first batch only)
            if do_grad_monitor and not grad_monitor_done:
                grad_monitor_metrics = _compute_grad_monitoring(agent, ppo_loss, aux_loss)
                grad_monitor_done = True

            loss.backward()

            # Clip norm + step
            if grad_clip_max_norm is not None and float(grad_clip_max_norm) > 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    agent.params, float(grad_clip_max_norm)
                )
            else:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    agent.params, float("inf")
                )
            total_grad_norm += float(grad_norm_tensor.item())
            n_optimizer_steps += 1
            agent.optimizer.step()

            # Accumulate metrics
            with torch.no_grad():
                total_loss += float(loss.item())
                total_ppo_loss += float(ppo_loss.item())
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy_loss += float(-entropy.item())
                total_aux_loss += float(aux_loss.item())

                # Approx KL: placeholder for future implementation
                approx_kl = 0.0

                # Clip fraction
                clipfrac = float(
                    (torch.abs(ratios - 1) > float(agent.clip_param)).float().mean().item()
                )
                total_clipfrac += clipfrac

                for k, v in aux_metrics.items():
                    aux_metrics_accum[k] = aux_metrics_accum.get(k, 0.0) + float(v)

            n_batches += 1

    agent.buffer.clear()

    current_lr = float(agent.optimizer.param_groups[0]["lr"])
    metrics = _finalize_metrics(
        total_loss=total_loss,
        total_ppo_loss=total_ppo_loss,
        total_policy_loss=total_policy_loss,
        total_value_loss=total_value_loss,
        total_entropy_loss=total_entropy_loss,
        total_aux_loss=total_aux_loss,
        total_approx_kl=total_approx_kl,
        total_clipfrac=total_clipfrac,
        total_grad_norm=total_grad_norm,
        aux_metrics_accum=aux_metrics_accum,
        n_batches=n_batches,
        n_optimizer_steps=n_optimizer_steps,
        current_lr=current_lr,
        batch_size=resolved_batch_size,
        grad_monitor_metrics=grad_monitor_metrics,
        grad_monitor_done=grad_monitor_done,
        logger=logger,
    )

    if logger is not None:
        step = int(getattr(agent, "_training_steps", 0))
        logger.log_update_metrics(metrics, step)
        logger.increment_update_count()

    return metrics


# ---------------------------------------------------------------------------
# MAPPO update (timestep-level batching with global-state critic)
# ---------------------------------------------------------------------------
def mappo_update(agent, aux_loss_fn=None) -> Dict[str, float]:
    batch = agent.buffer.get_all()
    if int(agent.buffer.ptr) <= 0:
        return {}

    logger = _get_logger(agent)
    do_grad_monitor = logger is not None and logger.should_monitor_grads()
    grad_monitor_metrics: Dict[str, float] = {}
    grad_monitor_done = False

    grad_clip_max_norm = getattr(agent, "grad_clip_max_norm", None)
    agent.extractor.train()
    agent.actor.train()
    agent.critic.train()
    _freeze_batchnorm_running_stats(agent.extractor)
    _freeze_batchnorm_running_stats(agent.actor)
    _freeze_batchnorm_running_stats(agent.critic)

    # --- Critic mode and data shapes ---
    critic_mode = getattr(agent, "critic_mode", "per_agent")
    T, N = batch["actions"].shape

    # Flatten per-agent data for extractor/actor
    obs_flat = _flatten_obs(batch["obs"])
    actions_flat = batch["actions"].reshape(-1)
    old_logprobs_flat = batch["logprobs"].reshape(-1)
    old_values_flat = batch["values"].reshape(-1)
    action_masks_flat = _flatten_2d(batch.get("action_masks"))
    use_fp_flat = _flatten_2d(batch.get("use_fp"))
    kpc_flat = _flatten_2d(batch.get("k_path_conflict"))

    # --- Returns and advantages ---
    if critic_mode == "team":
        team_rewards = batch["rewards"].sum(dim=1, keepdim=True)
        team_values = batch["values"][:, 0:1]
        team_dones = batch["dones"][:, 0:1]
        team_next_values = batch["next_values"][:, 0:1]
        returns = agent.compute_returns(team_rewards, team_values, team_dones, team_next_values)
        returns = returns.squeeze(1)                                    # (T,)
        advantages = returns - batch["values"][:, 0]                    # (T,)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_flat = advantages.unsqueeze(1).expand(T, N).reshape(-1).detach()
    else:
        returns = agent.compute_returns(batch["rewards"], batch["values"], batch["dones"], batch["next_values"])
        returns_flat = returns.reshape(-1)                              # (T*N,)
        advantages = (returns - batch["values"]).reshape(-1)            # (T*N,)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_flat = advantages.detach()

    # --- Timestep-level batching ---
    n_timesteps = T
    ts_batch_size = int(getattr(agent, "batch_size", 0) or 0)
    if ts_batch_size <= 0:
        ts_batch_size = n_timesteps
    else:
        ts_batch_size = min(ts_batch_size, n_timesteps)

    # Metric accumulators
    total_loss = 0.0
    total_ppo_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_aux_loss = 0.0
    total_approx_kl = 0.0
    total_clipfrac = 0.0
    total_grad_norm = 0.0
    aux_metrics_accum: Dict[str, float] = {}
    n_batches = 0
    n_optimizer_steps = 0

    arange_N = torch.arange(N, device=batch["actions"].device)

    for _ in range(int(agent.ppo_epoch)):
        ts_perm = torch.randperm(n_timesteps, device=batch["actions"].device)
        for ts_start in range(0, n_timesteps, ts_batch_size):
            ts_end = min(ts_start + ts_batch_size, n_timesteps)
            ts_idx = ts_perm[ts_start:ts_end]               # (B,)
            B = ts_idx.shape[0]
            flat_idx = (ts_idx.unsqueeze(1) * N + arange_N).reshape(-1)  # (B*N,)

            agent.optimizer.zero_grad()

            b_obs = _slice_obs(obs_flat, flat_idx)
            b_actions = actions_flat[flat_idx]
            b_old_logprobs = old_logprobs_flat[flat_idx]
            b_advantages = advantages_flat[flat_idx]
            b_action_masks = action_masks_flat[flat_idx] if action_masks_flat is not None else None

            # Extractor + Actor: per-agent (B*N, feature_dim)
            features = agent.extractor(b_obs)
            logits = agent.actor(features)
            if b_action_masks is not None:
                logits = logits.masked_fill(b_action_masks <= 0, -1e9)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(b_actions)
            entropies = dist.entropy()
            ratios = torch.exp(new_logprobs - b_old_logprobs)

            # Critic: group features for global-state view
            grouped = features.reshape(B, N, -1)             # (B, N, feature_dim)
            new_values = agent.critic(grouped)               # (B, N) or (B, 1)

            if critic_mode == "team":
                b_team_returns = returns[ts_idx]             # (B,)
                b_team_old_values = old_values_flat[ts_idx * N]  # (B,)
                value_loss = _compute_value_loss_mappo(
                    new_values.squeeze(-1), b_team_old_values, b_team_returns, agent,
                )
            else:
                b_returns = returns_flat[flat_idx]           # (B*N,)
                b_old_values = old_values_flat[flat_idx]     # (B*N,)
                value_loss = _compute_value_loss_mappo(
                    new_values.reshape(-1), b_old_values, b_returns, agent,
                )

            # Policy loss with sample filtering
            policy_mode = getattr(agent, "policy_update_mode", "all")
            policy_mask = _compute_policy_update_mask(policy_mode, use_fp_flat, kpc_flat, flat_idx)
            policy_loss = _compute_masked_policy_loss(ratios, b_advantages, float(agent.clip_param), policy_mask)

            entropy = entropies.mean()

            aux_loss, aux_metrics = _call_aux_loss(aux_loss_fn, batch, flat_idx, new_logprobs)
            ppo_loss = (
                policy_loss
                + float(agent.value_loss_coef) * value_loss
                - float(agent.entropy_coef) * entropy
            )
            loss = ppo_loss + aux_loss

            # Gradient monitoring (first batch only)
            if do_grad_monitor and not grad_monitor_done:
                grad_monitor_metrics = _compute_grad_monitoring(agent, ppo_loss, aux_loss)
                grad_monitor_done = True

            loss.backward()

            # Clip norm + step
            if grad_clip_max_norm is not None and float(grad_clip_max_norm) > 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    agent.params, float(grad_clip_max_norm)
                )
            else:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    agent.params, float("inf")
                )
            total_grad_norm += float(grad_norm_tensor.item())
            n_optimizer_steps += 1
            agent.optimizer.step()

            # Accumulate metrics
            with torch.no_grad():
                total_loss += float(loss.item())
                total_ppo_loss += float(ppo_loss.item())
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy_loss += float(-entropy.item())
                total_aux_loss += float(aux_loss.item())

                # Approx KL: placeholder for future implementation
                approx_kl = 0.0

                clipfrac = float(
                    (torch.abs(ratios - 1) > float(agent.clip_param)).float().mean().item()
                )
                total_clipfrac += clipfrac

                for k, v in aux_metrics.items():
                    aux_metrics_accum[k] = aux_metrics_accum.get(k, 0.0) + float(v)

            n_batches += 1

    agent.buffer.clear()

    resolved_batch_size = ts_batch_size * N
    current_lr = float(agent.optimizer.param_groups[0]["lr"])
    metrics = _finalize_metrics(
        total_loss=total_loss,
        total_ppo_loss=total_ppo_loss,
        total_policy_loss=total_policy_loss,
        total_value_loss=total_value_loss,
        total_entropy_loss=total_entropy_loss,
        total_aux_loss=total_aux_loss,
        total_approx_kl=total_approx_kl,
        total_clipfrac=total_clipfrac,
        total_grad_norm=total_grad_norm,
        aux_metrics_accum=aux_metrics_accum,
        n_batches=n_batches,
        n_optimizer_steps=n_optimizer_steps,
        current_lr=current_lr,
        batch_size=resolved_batch_size,
        grad_monitor_metrics=grad_monitor_metrics,
        grad_monitor_done=grad_monitor_done,
        logger=logger,
    )

    if logger is not None:
        step = int(getattr(agent, "_training_steps", 0))
        logger.log_update_metrics(metrics, step)
        logger.increment_update_count()

    return metrics

