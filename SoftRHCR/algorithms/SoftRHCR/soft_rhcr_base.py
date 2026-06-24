"""Shared logic for SoftRHCR agents (IPPO and MAPPO variants).

All SoftRHCR-specific behaviour that is independent of the critic
architecture lives here.  Concrete agents only need to implement:
  __init__, _rebuild_policy_modules, _planner_follow_action,
  select_action, update, load_model.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from SoftRHCR.modules.observation_adapter import SoftRHCRObservationAdapter
from SoftRHCR.modules.replay_buffer import _flatten_self_states
from SoftRHCR.modules.intrinsic_reward import compute_cr_intrinsic_rewards


class SoftRHCRBaseMixin:
    """Mixin holding SoftRHCR logic shared between IPPO and MAPPO agents.

    Must be used as the first base class alongside IPPOAgent or MAPPOAgent:
        class SoftRHCRAgent(SoftRHCRBaseMixin, IPPOAgent): ...
        class SoftRHCRMAPPOAgent(SoftRHCRBaseMixin, MAPPOAgent): ...
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_soft_rhcr_attributes(self, config: Any, obs_info, n_agents: int) -> None:
        """Set SoftRHCR-specific config and state attributes.

        Called from the concrete agent's ``__init__`` after
        ``super().__init__(...)`` and before ``_rebuild_policy_modules``.
        """
        self.reward_mode = getattr(config, "reward_mode", "legacy")
        self.soft_rhcr_L = getattr(config, "soft_rhcr_L", 2)
        self.soft_rhcr_k = getattr(config, "soft_rhcr_k", 10)
        self.force_rl_prob_start = getattr(config, "force_rl_prob_start", 0.8)
        self.force_rl_prob_end = getattr(config, "force_rl_prob_end", 0.2)
        self.decay_steps_ratio = getattr(config, "decay_steps_ratio", 0.5)
        self.soft_rhcr_msgs_mode = getattr(config, "soft_rhcr_msgs_mode", "dual")
        self.policy_update_mode = getattr(config, "policy_update_mode", "on_policy")
        self.planner_aux_loss = getattr(config, "planner_aux_loss", "consistency")
        self.kl_coef = getattr(config, "kl_coef", 0.0)
        self.kl_coef_end = getattr(config, "kl_coef_end", None)
        self.fp_consistency_coef = getattr(config, "fp_consistency_coef", 0.1)
        self.fp_consistency_coef_end = getattr(config, "fp_consistency_coef_end", None)
        self.fp_consistency_pmin = getattr(config, "fp_consistency_pmin", 0.6)
        self.fp_consistency_safe_only = getattr(config, "fp_consistency_safe_only", True)
        self.max_train_steps = int(getattr(config, "max_train_steps", 10**6))

        self.last_force_rl_prob = self.force_rl_prob_start
        self.last_fp_consistency_coef = self.fp_consistency_coef
        self.last_kl_coef = self.kl_coef
        self._training_steps = 0
        self._eval_skip_value = False

        self.obs_adapter = self.setup_observation_adapter(obs_info, n_agents, config=config)
        self.obs_shape = self.obs_adapter.compute_adapted_obs_shape(n_agents)

        self._observation_context: Dict[str, Any] = {}
        self._last_adapted_obs: Optional[Dict[str, Any]] = None
        self._last_obs_aux: Dict[str, Dict[str, Any]] = {}
        self._steps_to_follow: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Observation adapter
    # ------------------------------------------------------------------
    def setup_observation_adapter(self, obs_info, n_agents, config=None):
        return SoftRHCRObservationAdapter(obs_info, n_agents, config=config if config is not None else self)

    def _adapter_state(self) -> Dict[str, Any]:
        return {
            "env_info": self._observation_context,
            "planner_commit_remaining": dict(self._steps_to_follow),
        }

    def set_observation_context(self, info: Optional[Dict[str, Any]]) -> None:
        self._observation_context = dict(info or {})

    def adapt_observations(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        adapted = self.obs_adapter.adapt(obs, adapter_state=self._adapter_state())
        self._last_adapted_obs = adapted
        self._last_obs_aux = dict(getattr(self.obs_adapter, "last_aux", {}) or {})
        return adapted

    def get_last_adapted_observations(self) -> Optional[Dict[str, Any]]:
        return self._last_adapted_obs

    def get_observation_aux(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._last_obs_aux)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        self._steps_to_follow.clear()
        self._last_adapted_obs = None
        self._last_obs_aux = {}
        self.obs_adapter.reset_episode()

    def reset_done_agents(self, dones: Optional[Dict[str, Any]]) -> None:
        if isinstance(dones, dict):
            for aid, done in dones.items():
                if bool(done):
                    self._steps_to_follow[str(aid)] = 0
        self.obs_adapter.reset_done_agents(dones)

    # ------------------------------------------------------------------
    # Coefficient annealing
    # ------------------------------------------------------------------
    def _anneal_coef(self, start: float, end: Optional[float]) -> float:
        if end is None:
            return float(start)
        total_steps = max(1, int(self.max_train_steps))
        decay_steps = max(1.0, float(total_steps) * float(self.decay_steps_ratio))
        progress = min(1.0, float(self._training_steps) / float(decay_steps))
        return float(start) + (float(end) - float(start)) * float(progress)

    def get_force_rl_prob(self, step=None):
        if step is None:
            step = self._training_steps
        total_steps = max(1, int(getattr(self, "max_train_steps", 10**6)))
        decay_steps = max(1.0, float(total_steps) * float(self.decay_steps_ratio))
        progress = min(1.0, float(step) / float(decay_steps))
        prob = float(self.force_rl_prob_start) + (
            float(self.force_rl_prob_end) - float(self.force_rl_prob_start)
        ) * float(progress)
        lo = min(float(self.force_rl_prob_start), float(self.force_rl_prob_end))
        hi = max(float(self.force_rl_prob_start), float(self.force_rl_prob_end))
        return float(min(hi, max(lo, prob)))

    def _current_fp_consistency_coef(self) -> float:
        coef = self._anneal_coef(self.fp_consistency_coef, self.fp_consistency_coef_end)
        self.last_fp_consistency_coef = float(coef)
        return float(coef)

    def _current_kl_coef(self) -> float:
        coef = self._anneal_coef(self.kl_coef, self.kl_coef_end)
        self.last_kl_coef = float(coef)
        return float(coef)

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _agent_ids(obs: Dict[str, Any]) -> list[str]:
        return sorted(
            [str(aid) for aid in obs.keys()],
            key=lambda x: int(str(x).split("_")[-1]),
        )

    def _stack_obs(self, obs: Dict[str, Any], agent_ids: list[str]) -> Dict[str, torch.Tensor]:
        expected_ss_dim = int(self.buffer.obs["self_states"].shape[-1])
        ss_arrays = []
        for aid in agent_ids:
            ss_vec = np.asarray(_flatten_self_states(obs[aid]["self_states"]), dtype=np.float32)
            if int(ss_vec.size) != expected_ss_dim:
                raise RuntimeError(
                    f"_stack_obs self_states dim mismatch for {aid}: "
                    f"expected {expected_ss_dim}, got {int(ss_vec.size)}"
                )
            ss_arrays.append(ss_vec)
        return {
            "fov": torch.as_tensor(
                np.stack([np.asarray(obs[aid]["fov"], dtype=np.float32) for aid in agent_ids], axis=0),
                device=self.device,
            ),
            "msgs": torch.as_tensor(
                np.stack([np.asarray(obs[aid]["msgs"], dtype=np.float32) for aid in agent_ids], axis=0),
                device=self.device,
            ),
            "self_states": torch.as_tensor(
                np.stack(ss_arrays, axis=0),
                device=self.device,
            ),
        }

    def _stack_action_masks(self, action_masks: Optional[Dict[str, Any]], agent_ids: list[str]) -> Optional[torch.Tensor]:
        if not isinstance(action_masks, dict):
            return None
        return torch.as_tensor(
            np.stack(
                [
                    np.asarray(action_masks.get(aid, np.ones((self.n_actions,), dtype=np.float32)), dtype=np.float32)
                    for aid in agent_ids
                ],
                axis=0,
            ),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Action interface
    # ------------------------------------------------------------------
    def get_actions(self, obs, action_masks=None, **kwargs):
        actions, _, _, _ = self.select_action(
            obs,
            evaluation=bool(kwargs.get("evaluation", False)),
            action_masks=action_masks,
            info=kwargs.get("info"),
        )
        return actions

    # ------------------------------------------------------------------
    # Intrinsic rewards
    # ------------------------------------------------------------------
    def compute_intrinsic_rewards(
        self,
        next_obs,
        prev_obs_aux=None,
        next_obs_aux=None,
    ):
        """Compute SoftRHCR-specific intrinsic rewards.

        Reward shaping is driven by the change in ``k_path_conflict`` stored in
        ``obs_aux``. ``super()`` is intentionally not called so the MRO stays
        isolated — the parent's intrinsic reward is not added on top.
        """
        return compute_cr_intrinsic_rewards(
            next_obs=next_obs,
            prev_obs_aux=prev_obs_aux,
            next_obs_aux=next_obs_aux,
            reward_mode=self.reward_mode,
        )

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def set_training_mode(self, mode=True):
        self.extractor.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)

    def set_training_step(self, step):
        self._training_steps = int(step)

    def save_model(self, path, meta=None):
        payload = {
            "extractor": self.extractor.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": int(self._training_steps),
            "critic_mode": self.critic_mode,
        }
        if meta is not None:
            payload["meta"] = meta
        torch.save(payload, path)
