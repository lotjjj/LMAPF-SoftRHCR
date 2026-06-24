from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from SoftRHCR.algorithms.MAPPO.mappo import MAPPOAgent
from SoftRHCR.algorithms.SoftRHCR.soft_rhcr_base import SoftRHCRBaseMixin
from SoftRHCR.modules.device import try_load_optimizer_state
from SoftRHCR.modules.network import Actor, FeatureExtractor, MAPPOCritic
from SoftRHCR.modules.planner_features import (
    extract_shared_planner_paths,
    path_array_from_entry,
)
from SoftRHCR.modules.replay_buffer import PPOBuffer


class SoftRHCRMAPPOAgent(SoftRHCRBaseMixin, MAPPOAgent):
    def __init__(self, config, obs_info, action_dim, n_agents, n_actions=None):
        super().__init__(config, obs_info, action_dim, n_agents, n_actions=n_actions)
        self._init_soft_rhcr_attributes(config, obs_info, n_agents)
        self._rebuild_policy_modules(config, self.obs_shape)

    def _rebuild_policy_modules(self, config: Any, obs_shape: Dict[str, Any]) -> None:
        self.extractor = FeatureExtractor(
            fov_shape=tuple(obs_shape.get("fov", (5, 11, 11))),
            msg_shape=tuple(obs_shape.get("msgs", (10, 11, 11))),
            self_state_dim=int(obs_shape.get("self_states", 25)),
            feature_dim=self.feature_dim,
            norm_type=getattr(config, "norm_type", "gn"),
            backbone=getattr(config, "extractor_backbone", "default"),
            norm_after_concat=getattr(config, "norm_after_concat", "none"),
            msgs_mode=getattr(config, "soft_rhcr_msgs_mode", "dual"),
        ).to(self.device)
        self.actor = Actor(self.feature_dim, self.n_actions).to(self.device)
        self.critic = MAPPOCritic(self.feature_dim, self.n_agents, mode=self.critic_mode).to(self.device)
        self.params = (
            list(self.extractor.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
        self.optimizer = self._optimizer_cls(self.params, lr=self.lr, weight_decay=self.weight_decay)
        self.buffer = PPOBuffer(
            n_steps=self.horizon_len,
            n_agents=self.n_agents,
            obs_shape=obs_shape,
            action_dim=self.n_actions,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Planner following (MAPPO variant – returns path metadata)
    # ------------------------------------------------------------------
    def _planner_follow_action(self, agent_id: str, raw_obs: Dict[str, Any]) -> tuple[int, str, Dict[str, bool]]:
        planner_paths = extract_shared_planner_paths(self._observation_context)
        entry = planner_paths.get(agent_id) if isinstance(planner_paths, dict) else None
        if not isinstance(entry, dict):
            return 4, "missing_entry", {"path0_matches_current": False, "path1_matches_current": False}
        if not bool(entry.get("has_path", False)):
            return 4, "has_path_false", {"path0_matches_current": False, "path1_matches_current": False}
        agent_obs = raw_obs.get(agent_id)
        if not isinstance(agent_obs, dict):
            return 4, "missing_obs", {"path0_matches_current": False, "path1_matches_current": False}
        # Get current position from path[0] directly
        seq = path_array_from_entry(entry, fallback_pos=(0, 0))
        cur_x = int(round(float(seq[0][0])))
        cur_y = int(round(float(seq[0][1])))
        if seq.shape[0] <= 1:
            return 4, "short_path", {"path0_matches_current": False, "path1_matches_current": False}
        rounded_seq = np.rint(seq).astype(np.int32, copy=False)
        next_x = int(rounded_seq[1][0])
        next_y = int(rounded_seq[1][1])
        meta = {
            "path0_matches_current": True,  # path[0] IS the current position now
            "path1_matches_current": bool(next_x == cur_x and next_y == cur_y),
        }
        dx = int(next_x - cur_x)
        dy = int(next_y - cur_y)
        if dx == 0 and dy == -1:
            return 0, "move_up", meta
        if dx == 0 and dy == 1:
            return 1, "move_down", meta
        if dx == -1 and dy == 0:
            return 2, "move_left", meta
        if dx == 1 and dy == 0:
            return 3, "move_right", meta
        if dx == 0 and dy == 0:
            return 4, "same_cell", meta
        return 4, "non_cardinal", meta

    # ------------------------------------------------------------------
    # Action selection (MAPPO variant – two-phase gate, detailed stats)
    # ------------------------------------------------------------------
    def select_action(self, obs, evaluation=False, action_masks=None, info=None):
        if info is not None:
            self.set_observation_context(info)
        raw_obs = obs

        # Phase 1: Compute gate context (KPC, planner paths)
        gate_ctx = self.obs_adapter.compute_gate_context(raw_obs, adapter_state=self._adapter_state())
        pre_gate_aux = gate_ctx["aux_base"]
        agent_ids = sorted(
            [str(aid) for aid in gate_ctx.get("agent_ids", [])],
            key=lambda x: int(str(x).split("_")[-1]),
        )

        force_rl_prob = 0.0 if evaluation else self.get_force_rl_prob()
        self.last_force_rl_prob = float(force_rl_prob)

        use_fp_dict: Dict[str, bool] = {}
        fp_actions: Dict[str, int] = {}
        gate_reason_counts = {
            "follow_commit": 0,
            "kpc_zero_new_follow": 0,
            "kpc_zero_forced_rl": 0,
            "kpc_positive_rl": 0,
            "no_path_rl": 0,
            "path_lost_rl": 0,
        }
        fp_action_reason_counts = {
            "missing_obs": 0,
            "missing_entry": 0,
            "has_path_false": 0,
            "short_path": 0,
            "same_cell": 0,
            "non_cardinal": 0,
            "move_up": 0,
            "move_down": 0,
            "move_left": 0,
            "move_right": 0,
        }
        fp_selected_count = 0
        fp_wait_count = 0
        fp_path0_mismatch_count = 0
        fp_path1_matches_current_count = 0

        # Phase 1.5: Clear _steps_to_follow for completed agents + compute FP actions
        fp_action_infos: Dict[str, tuple] = {}
        for aid in agent_ids:
            aux = pre_gate_aux.get(aid, {}) if isinstance(pre_gate_aux, dict) else {}
            if bool(aux.get("task_completed", False)):
                self._steps_to_follow[aid] = 0
            fp_action, fp_action_reason, fp_action_meta = self._planner_follow_action(aid, raw_obs)
            fp_actions[aid] = int(fp_action)
            fp_action_infos[aid] = (fp_action_reason, fp_action_meta)

        # Phase 2: Gate logic + state updates
        _VALID_FP_REASONS = {"move_up", "move_down", "move_left", "move_right", "same_cell"}
        for aid in agent_ids:
            aux = pre_gate_aux.get(aid, {}) if isinstance(pre_gate_aux, dict) else {}

            # Check if planner has a valid actionable path for this agent
            fp_reason = fp_action_infos[aid][0] if aid in fp_action_infos else ""
            has_valid_path = fp_reason in _VALID_FP_REASONS

            follow_remaining = int(self._steps_to_follow.get(aid, 0))
            if not has_valid_path:
                # No valid planner path → force RL, clear follow commitment
                use_fp = False
                self._steps_to_follow[aid] = 0
                if follow_remaining > 0:
                    gate_reason_counts["path_lost_rl"] += 1
                else:
                    gate_reason_counts["no_path_rl"] += 1
            elif follow_remaining > 0:
                use_fp = True
                gate_reason_counts["follow_commit"] += 1
            else:
                kpc = int(aux.get("k_path_conflict", 0))
                if kpc == 0:
                    force_to_rl = (not evaluation) and (np.random.random() < float(force_rl_prob))
                    if force_to_rl:
                        use_fp = False
                        gate_reason_counts["kpc_zero_forced_rl"] += 1
                    else:
                        use_fp = True
                        gate_reason_counts["kpc_zero_new_follow"] += 1
                        self._steps_to_follow[aid] = max(1, int(self.soft_rhcr_L))
                else:
                    use_fp = False
                    gate_reason_counts["kpc_positive_rl"] += 1

            use_fp_dict[aid] = bool(use_fp)
            if use_fp_dict[aid]:
                fp_selected_count += 1
                fp_action = int(fp_actions[aid])
                fp_action_reason, fp_action_meta = fp_action_infos[aid]
                if fp_action == 4:
                    fp_wait_count += 1
                fp_action_reason_counts[fp_action_reason] = fp_action_reason_counts.get(fp_action_reason, 0) + 1
                if not bool(fp_action_meta.get("path0_matches_current", False)):
                    fp_path0_mismatch_count += 1
                if bool(fp_action_meta.get("path1_matches_current", False)):
                    fp_path1_matches_current_count += 1
                self._steps_to_follow[aid] = max(0, int(self._steps_to_follow.get(aid, 0)) - 1)

        n_gate = max(1, len(agent_ids))

        # Phase 3: Build observation (post-gate, committed info is precise)
        policy_obs = self.obs_adapter.build_policy_observation(raw_obs, gate_ctx, committed_agents=use_fp_dict)
        self._last_adapted_obs = policy_obs
        self._last_obs_aux = self.obs_adapter.last_aux

        # Phase 4: Forward pass
        obs_tensor = self._stack_obs(policy_obs, agent_ids)
        action_mask_tensor = self._stack_action_masks(action_masks, agent_ids)

        with torch.no_grad():
            features = self.extractor(obs_tensor)
            logits = self.actor(features)
            if action_mask_tensor is not None:
                logits = logits.masked_fill(action_mask_tensor <= 0, -1e9)
            dist = Categorical(logits=logits)
            if evaluation:
                rl_actions = torch.argmax(logits, dim=-1)
            else:
                rl_actions = dist.sample()
            rl_logprobs = dist.log_prob(rl_actions)
            if evaluation and bool(getattr(self, "_eval_skip_value", False)):
                values_tensor = torch.zeros((len(agent_ids),), device=self.device, dtype=torch.float32)
            else:
                values_tensor = self._compute_values_from_features(features)

        fp_action_tensor = torch.as_tensor(
            [int(fp_actions[aid]) for aid in agent_ids],
            device=self.device,
            dtype=torch.long,
        )
        fp_logprobs = dist.log_prob(fp_action_tensor)

        actions: Dict[str, int] = {}
        logprobs: Dict[str, float] = {}
        values: Dict[str, float] = {}
        for idx, aid in enumerate(agent_ids):
            values[aid] = float(values_tensor[idx].detach().cpu().item())
            if bool(use_fp_dict[aid]):
                actions[aid] = int(fp_actions[aid])
                logprobs[aid] = float(fp_logprobs[idx].detach().cpu().item())
            else:
                actions[aid] = int(rl_actions[idx].detach().cpu().item())
                logprobs[aid] = float(rl_logprobs[idx].detach().cpu().item())

        return actions, logprobs, values, use_fp_dict

    def update(self):
        from SoftRHCR.algorithms.PPO.ppo_update import mappo_update
        from SoftRHCR.algorithms.SoftRHCR.rhcr_utils import make_soft_rhcr_aux_loss

        return mappo_update(self, aux_loss_fn=make_soft_rhcr_aux_loss(self))

    def load_model(self, path, load_critic=True, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.extractor.load_state_dict(checkpoint["extractor"])
        self.actor.load_state_dict(checkpoint["actor"])
        if load_critic and "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer and "optimizer" in checkpoint:
            try_load_optimizer_state(self.optimizer, checkpoint["optimizer"], self.device)
        self._training_steps = int(checkpoint.get("training_steps", 0))
