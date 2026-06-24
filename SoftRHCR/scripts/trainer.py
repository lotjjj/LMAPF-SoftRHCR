import json
import time
import os
import random
import math
import gc
import torch
import numpy as np
from dataclasses import asdict
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from LMAPFEnv import WarehouseEnv
from SoftRHCR.config.configBase import EnvConfig, CommonConfig, RunConfig
from SoftRHCR.config.config_loader import load_and_build_config
from SoftRHCR.config.registry import algorithm_name_from_agent, create_agent as registry_create_agent
from SoftRHCR.modules.device import resolve_device, manual_seed_all
from SoftRHCR.modules.model_io import load_training_checkpoint
from SoftRHCR.scripts.cli_review import confirm_resolved_config
from SoftRHCR.modules.train_logger import TrainLogger


def set_global_seeds(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manual_seed_all(device, seed)

class Trainer:
    def __init__(self, env_cfg: EnvConfig, algo_cfg: CommonConfig):
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.device = resolve_device(getattr(algo_cfg, "device", "auto"))
        self.algo_cfg.device = str(self.device)
        set_global_seeds(self.env_cfg.train.seed, self.device)
        self._stop_requested = False
        self._interrupt_checkpoint_saved = False
        self.last_interrupt_checkpoint_path: Optional[str] = None
        self.last_interrupt_total_steps: int = 0
        self.last_interrupt_episode: int = 0
        self._lr0 = float(getattr(self.algo_cfg, "lr", 0.0))
        self._min_lr = float(getattr(self.algo_cfg, "min_lr", 0.0))
        raw_sched = getattr(self.algo_cfg, "lr_schedule", None)
        self._lr_schedule = str(raw_sched).strip().lower() if raw_sched is not None else "none"
        if self._lr_schedule in ("", "none", "null"):
            self._lr_schedule = "none"
        if self._lr_schedule not in ("none", "linear", "cosine"):
            raise ValueError(f"Unsupported lr_schedule={raw_sched}, expected linear/cosine/none")

        try:
            from SoftRHCR.config.configBase import SoftRHCRConfig, SoftRHCRMAPPOConfig
            if isinstance(self.algo_cfg, (SoftRHCRConfig, SoftRHCRMAPPOConfig)):
                from SoftRHCR.algorithms.SoftRHCR.rhcr_utils import apply_soft_rhcr_defaults
                apply_soft_rhcr_defaults(self.env_cfg, self.algo_cfg)
        except Exception:
            pass

        self.train_env = WarehouseEnv(**env_cfg.get_env_args(mode="train"))
        self._train_render_interval_s = max(0.0, float(getattr(self.env_cfg.train, "render_interval_s", 0.0) or 0.0))

        self.n_agents = len(self.train_env.possible_agents)
    
        obs_info = self._get_obs_info(self.train_env)
        action_dim = self.train_env.action_spaces['agv_0'].n
        
        self.agent = self._create_agent(algo_cfg, obs_info, action_dim, self.n_agents)
        self.observation_snapshot = self._build_observation_snapshot(self.train_env, obs_info)

        # Initialise unified TensorBoard logger → run_dir/run_name/log
        _model_dir = getattr(self.algo_cfg, "model_dir", None)
        if _model_dir is not None:
            _root = os.path.dirname(str(_model_dir))
        else:
            _root = os.getcwd()
        self.logger: Optional[TrainLogger] = TrainLogger(log_dir=os.path.join(_root, "log"))
        self.agent.logger = self.logger

        # Episode-level metric accumulators (train/ prefixed metrics)
        self._episode_reward_accum: float = 0.0
        self._episode_steps: int = 0
        self._episode_task_count: int = 0
        self._episode_conflict_sum: float = 0.0
        self._episode_use_fp_count: int = 0
        self._episode_agent_steps: int = 0

    def _set_optimizer_lr(self, lr: float) -> None:
        if lr <= 0.0:
            return

        opt_base = getattr(torch.optim, "Optimizer", None)
        if opt_base is None:
            return

        def _apply(obj) -> None:
            if isinstance(obj, opt_base):
                for group in obj.param_groups:
                    group["lr"] = float(lr)
                return
            if isinstance(obj, dict):
                for v in obj.values():
                    _apply(v)
                return
            if isinstance(obj, (list, tuple, set)):
                for v in obj:
                    _apply(v)
                return

        for v in getattr(self, "agent").__dict__.values():
            _apply(v)

    def _compute_scheduled_lr(self, total_steps: int, target_total_steps: int) -> Optional[float]:
        if self._lr_schedule == "none" or self._lr0 <= 0.0:
            return None
        progress = min(1.0, float(total_steps) / float(max(1, int(target_total_steps))))
        if self._lr_schedule == "linear":
            lr = float(self._lr0) * (1.0 - progress)
        else:  # cosine
            lr = float(self._min_lr) + (float(self._lr0) - float(self._min_lr)) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(float(self._min_lr), float(lr))

    def request_stop(self) -> None:
        self._stop_requested = True

    def _check_stop(self) -> None:
        if bool(getattr(self, "_stop_requested", False)):
            raise KeyboardInterrupt

    @staticmethod
    def _looks_like_oom(exc: BaseException) -> bool:
        text = str(exc).lower()
        return (
            isinstance(exc, MemoryError)
            or "out of memory" in text
            or "cuda out of memory" in text
            or "not enough memory" in text
            or "insufficient memory" in text
        )

    def _clear_memory(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def close(self) -> None:
        try:
            if hasattr(self, "train_env") and self.train_env is not None:
                self.train_env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Episode-level metric tracking helpers
    # ------------------------------------------------------------------
    def _accumulate_step_metrics(
        self,
        rewards: Dict[str, Any],
        next_obs: Dict[str, Any],
        next_info: Optional[Dict[str, Any]],
        use_fp_dict: Optional[Dict[str, Any]],
    ) -> None:
        """Accumulate per-step environment interaction metrics for the current episode."""
        step_reward = sum(float(r) for r in rewards.values())
        self._episode_reward_accum += step_reward
        self._episode_steps += 1

        n_agents_step = max(1, len(next_obs))
        n_conflicted = 0
        if isinstance(next_info, dict):
            for aid in next_obs.keys():
                v = next_info.get(aid) if isinstance(next_info.get(aid), dict) else None
                if v is not None and bool(v.get("task_completed", False)):
                    self._episode_task_count += 1
                if v is not None and bool(v.get("conflicted", False)):
                    n_conflicted += 1
        self._episode_conflict_sum += float(n_conflicted) / float(n_agents_step)

        if use_fp_dict is not None:
            n_use_fp = sum(1 for v in use_fp_dict.values() if bool(v))
            self._episode_use_fp_count += n_use_fp
            self._episode_agent_steps += n_agents_step

    def _log_episode_metrics(self, total_steps: int) -> None:
        """Log accumulated episode metrics to TensorBoard."""
        if self.logger is None:
            return
        ep_steps = max(1, self._episode_steps)
        n_agents = max(1, self.n_agents)
        agent_steps = max(1, self._episode_agent_steps)

        train_metrics = {
            "train/episode_reward": float(self._episode_reward_accum),
            "train/mean_step_reward": float(self._episode_reward_accum) / float(ep_steps * n_agents),
            "train/episode_task_completion": int(self._episode_task_count),
            "train/conflict_rate": float(self._episode_conflict_sum) / float(ep_steps),
            "train/use_fp_rate": float(self._episode_use_fp_count) / float(agent_steps),
            "train/force_rl_prob": float(getattr(self.agent, "last_force_rl_prob", 0.0)),
            "train/fp_consistency_coef": float(getattr(self.agent, "last_fp_consistency_coef", 0.0)),
            "train/kl_coef": float(getattr(self.agent, "last_kl_coef", 0.0)),
        }
        self.logger.log_train_metrics(train_metrics, int(total_steps))

    def _reset_episode_accumulators(self) -> None:
        """Reset all per-episode metric accumulators."""
        self._episode_reward_accum = 0.0
        self._episode_steps = 0
        self._episode_task_count = 0
        self._episode_conflict_sum = 0.0
        self._episode_use_fp_count = 0
        self._episode_agent_steps = 0

    def _get_obs_info(self, env):
        base_env = getattr(env, "_env", env)
        agent_id = env.possible_agents[0]
        obs_space = env.observation_spaces[agent_id]
        self_state_dim = 0
        if hasattr(obs_space, "spaces") and "self_states" in obs_space.spaces:
            for sub in obs_space.spaces["self_states"].spaces.values():
                if hasattr(sub, "shape"):
                    self_state_dim += int(np.prod(sub.shape))

        fov_shape = obs_space.spaces['fov'].shape
        if hasattr(obs_space, "spaces") and "msgs" in obs_space.spaces:
            msg_shape = obs_space.spaces['msgs'].shape
        else:
            msg_shape = (int(getattr(base_env, "kstep_conflict_check", 0)), int(fov_shape[1]), int(fov_shape[2]))
        return {
            'fov': fov_shape,
            'msgs': msg_shape,
            'self_states': self_state_dim,
            'map_width': int(getattr(base_env, "width", 1)),
            'map_height': int(getattr(base_env, "height", 1)),
        }

    def _build_observation_snapshot(self, env, raw_obs_info: Dict[str, Any]) -> Dict[str, Any]:
        agent_id = env.possible_agents[0]
        obs_space = env.observation_spaces[agent_id]
        raw_top_level_keys = list(obs_space.spaces.keys()) if hasattr(obs_space, "spaces") else []
        raw_self_state_keys: List[str] = []
        if hasattr(obs_space, "spaces") and "self_states" in obs_space.spaces:
            raw_self_spaces = getattr(obs_space.spaces["self_states"], "spaces", {})
            raw_self_state_keys = [str(key) for key in raw_self_spaces.keys()]

        policy_obs_shape = dict(getattr(self.agent, "obs_shape", {}) or {})
        adapter = getattr(self.agent, "obs_adapter", None)
        adapter_name = type(adapter).__name__ if adapter is not None else ""
        policy_self_state_keys = list(raw_self_state_keys)
        policy_top_level_keys = ["fov", "msgs", "self_states"]
        if adapter_name in ("DefaultObservationAdapter", "SoftRHCRObservationAdapter"):
            policy_self_state_keys = ["path_info", *policy_self_state_keys]

        return {
            "adapter": adapter_name,
            "msgs_mode": str(getattr(self.algo_cfg, "soft_rhcr_msgs_mode", "single") or "single"),
            "raw": {
                "top_level_keys": raw_top_level_keys,
                "self_state_keys": raw_self_state_keys,
                "shapes": {
                    "fov": tuple(raw_obs_info.get("fov", ())),
                    "self_states": int(raw_obs_info.get("self_states", 0)),
                },
                "derived_shapes": {
                    "planner_msgs_base": tuple(raw_obs_info.get("msgs", ())),
                },
            },
            "policy": {
                "top_level_keys": policy_top_level_keys,
                "self_state_keys": policy_self_state_keys,
                "shapes": {
                    "fov": tuple(policy_obs_shape.get("fov", ())),
                    "msgs": tuple(policy_obs_shape.get("msgs", ())),
                    "self_states": int(policy_obs_shape.get("self_states", 0) or 0),
                },
            },
        }

    def _create_agent(self, config, obs_info, action_dim, n_agents: int):
        """Algorithm factory."""
        return registry_create_agent(config, obs_info, action_dim, n_agents)

    def _algo_name(self) -> str:
        return algorithm_name_from_agent(self.agent)

    def _resolve_train_schedule(self) -> Tuple[int, int, int]:
        horizon_len = self.algo_cfg.resolved_horizon_len()
        target_total_steps = self.algo_cfg.resolved_target_total_steps()
        save_interval_steps = self.algo_cfg.resolved_save_interval_steps()
        return horizon_len, target_total_steps, save_interval_steps

    def _reset_train_episode(self, episode: int):
        env_seed = self.env_cfg.train.seed + int(episode)
        obs, info = self.train_env.reset(seed=env_seed)
        if hasattr(self.agent, "reset_episode"):
            self.agent.reset_episode()
        return obs, info, env_seed

    def _get_action_masks(self, env, obs: Dict[str, Any], info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if isinstance(info, dict):
            masks = {}
            for aid in obs.keys():
                v = info.get(aid) if isinstance(info.get(aid), dict) else None
                if v is not None and "action_mask" in v:
                    masks[aid] = v["action_mask"]
            if len(masks) > 0:
                return masks
        if hasattr(env, "action_mask"):
            try:
                return {aid: env.action_mask(aid) for aid in obs.keys()}
            except Exception:
                return None
        return None

    def _collect_step(self, obs, info):
        action_masks = self._get_action_masks(self.train_env, obs, info)
        if hasattr(self.agent, "set_observation_context"):
            self.agent.set_observation_context(info)
        use_fp_dict = None
        is_mappo_like = False
        agent_name = type(self.agent).__name__
        ppo_like_agents = {"IPPOAgent", "MAPPOAgent", "GateBlendAgent", "GateBlendMAPPOAgent"}
        info_select_agents = {"SoftRHCRAgent", "SoftRHCRMAPPOAgent"}
        if agent_name in ppo_like_agents or agent_name in info_select_agents:
            is_mappo_like = agent_name in {"MAPPOAgent", "GateBlendMAPPOAgent", "SoftRHCRMAPPOAgent"}
            if agent_name in info_select_agents:
                actions, logprobs, values, use_fp_dict = self.agent.select_action(
                    obs,
                    action_masks=action_masks,
                    info=info,
                )
            else:
                actions, logprobs, values = self.agent.select_action(obs, action_masks=action_masks)
        else:
            actions = self.agent.select_action(obs, action_masks=action_masks)
            logprobs = None
            values = None
        policy_obs = self.agent.get_last_adapted_observations() if hasattr(self.agent, "get_last_adapted_observations") else obs
        obs_aux = self.agent.get_observation_aux() if hasattr(self.agent, "get_observation_aux") else {}

        next_obs, rewards, terminations, truncations, next_info = self.train_env.step(actions)
        dones = {k: terminations[k] or truncations[k] for k in terminations}
        next_action_masks = self._get_action_masks(self.train_env, next_obs, next_info)
        if hasattr(self.agent, "set_observation_context"):
            self.agent.set_observation_context(next_info)
        # For intrinsic rewards, only KPC from next_obs is needed
        next_obs_aux = {}
        if hasattr(self.agent, "obs_adapter") and hasattr(self.agent.obs_adapter, "compute_gate_context"):
            next_gate_ctx = self.agent.obs_adapter.compute_gate_context(
                next_obs, adapter_state=self.agent._adapter_state()
            )
            next_obs_aux = next_gate_ctx.get("aux_base", {})
        elif hasattr(self.agent, "adapt_observations"):
            self.agent.adapt_observations(next_obs)
            next_obs_aux = self.agent.get_observation_aux() if hasattr(self.agent, "get_observation_aux") else {}
        # policy_next_obs for buffer storage (PPO update doesn't use next_obs data directly)
        if hasattr(self.agent, "obs_adapter") and hasattr(self.agent.obs_adapter, "compute_gate_context"):
            # SoftRHCR path: build adapted next_obs with msgs for buffer storage
            _committed = use_fp_dict if use_fp_dict is not None else {}
            policy_next_obs = self.agent.obs_adapter.build_policy_observation(
                next_obs, next_gate_ctx, committed_agents=_committed,
            )
        else:
            policy_next_obs = self.agent.adapt_observations(next_obs) if hasattr(self.agent, "adapt_observations") else next_obs
        intrinsic_rewards = self.agent.compute_intrinsic_rewards(
            next_obs,
            prev_obs_aux=obs_aux,
            next_obs_aux=next_obs_aux,
        )
        if isinstance(intrinsic_rewards, dict) and len(intrinsic_rewards) > 0:
            rewards = {aid: float(rewards.get(aid, 0.0)) + float(intrinsic_rewards.get(aid, 0.0)) for aid in next_obs.keys()}
        if hasattr(self.agent, "reset_done_agents"):
            self.agent.reset_done_agents(dones)

        if agent_name in ppo_like_agents or agent_name in info_select_agents:
            k_path_conflict = None
            conflicted = None
            if isinstance(obs_aux, dict) and len(obs_aux) > 0:
                k_path_conflict = {}
                conflicted = {}
                for aid in policy_obs.keys():
                    aux = obs_aux.get(aid) if isinstance(obs_aux.get(aid), dict) else {}
                    k_path_conflict[aid] = int(aux.get("k_path_conflict", 0))
                    v = next_info.get(aid) if isinstance(next_info.get(aid), dict) else None
                    conflicted[aid] = 1 if (v is not None and bool(v.get("conflicted", False))) else 0
            elif isinstance(next_info, dict):
                k_path_conflict = {}
                conflicted = {}
                for aid in policy_obs.keys():
                    k_path_conflict[aid] = 0
                    v = next_info.get(aid) if isinstance(next_info.get(aid), dict) else None
                    conflicted[aid] = 1 if (v is not None and bool(v.get("conflicted", False))) else 0
            task_completed = None
            if isinstance(next_info, dict):
                task_completed = {}
                for aid in policy_next_obs.keys():
                    v = next_info.get(aid) if isinstance(next_info.get(aid), dict) else None
                    task_completed[aid] = 1 if (v is not None and bool(v.get("task_completed", False))) else 0
            self.agent.buffer.add(
                policy_obs,
                actions,
                logprobs,
                rewards,
                values,
                dones,
                policy_next_obs,
                action_masks=action_masks,
                next_action_masks=next_action_masks,
                k_path_conflict=k_path_conflict,
                task_completed=task_completed,
                conflicted=conflicted,
                use_fp=use_fp_dict,
            )
        else:
            self.agent.buffer.add(policy_obs, actions, rewards, policy_next_obs, dones, action_masks=action_masks, next_action_masks=next_action_masks)

        return next_obs, rewards, dones, next_info, use_fp_dict, obs_aux, next_obs_aux

    def _save_checkpoint(
        self,
        total_steps: int,
        episode: int,
        env_seed: int,
        horizon_len: int,
        max_train_steps: int,
        target_total_steps: int,
        interrupted: bool = False,
    ) -> Optional[str]:
        if type(self.agent).__name__ in ("FollowPlannerAgent",):
            return None
        resume_episode = int(episode)
        resume_total_steps = int(total_steps)
        if interrupted:
            resume_episode = int(getattr(self, "_last_safe_episode", episode))
            resume_total_steps = int(getattr(self, "_last_safe_total_steps", total_steps))
        os.makedirs(self.algo_cfg.model_dir, exist_ok=True)
        ckpt_path = os.path.join(self.algo_cfg.model_dir, f"{type(self.agent).__name__}_step{total_steps}.pth")
        run_cfg_snapshot = RunConfig(
            algorithm=self._algo_name(),
            env=self.env_cfg,
            algo=self.algo_cfg,
            observation=dict(getattr(self, "observation_snapshot", {}) or {}),
        ).to_dict()
        self.agent.save_model(
            ckpt_path,
            meta={
                "episode": int(resume_episode),
                "total_steps": int(resume_total_steps),
                "env_seed": int(env_seed),
                "agent_class": type(self.agent).__name__,
                "algo_cfg_class": type(self.algo_cfg).__name__,
                "horizon_len": int(horizon_len),
                "max_train_steps": int(max_train_steps),
                "target_total_steps": int(target_total_steps),
                "interrupted": bool(interrupted),
                "interrupted_snapshot_episode": int(episode),
                "interrupted_snapshot_total_steps": int(total_steps),
                "algorithm": self._algo_name(),
                "algo_cfg": asdict(self.algo_cfg),
                "run_config": run_cfg_snapshot,
            },
        )
        return ckpt_path

    def _save_interrupt_checkpoint(
        self,
        total_steps: int,
        episode: int,
        env_seed: int,
        horizon_len: int,
        max_train_steps: int,
        target_total_steps: int,
    ) -> Optional[str]:
        if bool(getattr(self, "_interrupt_checkpoint_saved", False)):
            return getattr(self, "last_interrupt_checkpoint_path", None)
        ckpt_path = self._save_checkpoint(
            total_steps=int(total_steps),
            episode=int(episode),
            env_seed=int(env_seed),
            horizon_len=int(horizon_len),
            max_train_steps=int(max_train_steps),
            target_total_steps=int(target_total_steps),
            interrupted=True,
        )
        self._interrupt_checkpoint_saved = True
        self.last_interrupt_checkpoint_path = ckpt_path
        self.last_interrupt_total_steps = int(total_steps)
        self.last_interrupt_episode = int(episode)
        return ckpt_path

    def _maybe_save(self, total_steps: int, next_save_step: Optional[int], save_interval_steps: int, episode: int, env_seed: int, horizon_len: int, max_train_steps: int, target_total_steps: int) -> Optional[int]:
        if next_save_step is None or total_steps < next_save_step:
            return next_save_step
        self._save_checkpoint(
            total_steps=int(total_steps),
            episode=int(episode),
            env_seed=int(env_seed),
            horizon_len=int(horizon_len),
            max_train_steps=int(max_train_steps),
            target_total_steps=int(target_total_steps),
            interrupted=False,
        )
        return int(next_save_step + save_interval_steps)

    def run(self, start_episode: int = 0, total_steps: int = 0):
        episode = int(start_episode)
        env_seed = int(getattr(self.env_cfg.train, "seed", 0))
        tqdm_pos = 0
        try:
            tqdm_pos = int(os.environ.get("LMAPF_TQDM_POS", "0"))
        except Exception:
            tqdm_pos = 0
        tqdm_desc = str(getattr(self.algo_cfg, "tqdm_desc", "Training"))
        try:
            if hasattr(self.agent, "set_training_mode"):
                self.agent.set_training_mode(True)
            horizon_len, target_total_steps, save_interval_steps = self._resolve_train_schedule()
            max_train_steps = int(self.algo_cfg.max_train_steps)
            if int(total_steps) >= int(target_total_steps):
                with tqdm(
                    total=int(target_total_steps),
                    initial=min(int(total_steps), int(target_total_steps)),
                    desc=tqdm_desc,
                    unit="steps",
                    dynamic_ncols=True,
                    mininterval=0.2,
                    maxinterval=1.0,
                    position=tqdm_pos,
                    leave=True,
                ) as pbar:
                    pbar.set_postfix_str("target steps reached", refresh=True)
                return
            lr0 = self._compute_scheduled_lr(int(total_steps), int(target_total_steps))
            if lr0 is not None:
                self._set_optimizer_lr(lr0)

            obs = None
            info = None
            self._last_safe_episode = int(start_episode)
            self._last_safe_total_steps = int(total_steps)
            next_save_step = int(total_steps + save_interval_steps) if save_interval_steps > 0 else None

            with tqdm(
                total=int(target_total_steps),
                initial=int(total_steps),
                desc=tqdm_desc,
                unit="steps",
                dynamic_ncols=True,
                mininterval=0.2,
                maxinterval=1.0,
                position=tqdm_pos,
                leave=True,
            ) as pbar:
                while total_steps < target_total_steps:
                    self._check_stop()
                    horizon_steps = 0

                    if hasattr(self.agent, "set_training_mode"):
                        self.agent.set_training_mode(False)

                    while horizon_steps < horizon_len and total_steps < target_total_steps:
                        self._check_stop()
                        if obs is None:
                            obs, info, env_seed = self._reset_train_episode(episode)
                        if hasattr(self.agent, "set_training_step"):
                            self.agent.set_training_step(int(total_steps))
                        next_obs, rewards, dones, next_info, use_fp_dict, obs_aux, next_obs_aux = self._collect_step(obs, info)
                        if self.env_cfg.train.render_mode is not None and self._train_render_interval_s > 0.0:
                            time.sleep(self._train_render_interval_s)

                        # --- Accumulate episode-level env metrics ---
                        self._accumulate_step_metrics(rewards, next_obs, next_info, use_fp_dict)

                        total_steps += 1
                        horizon_steps += 1

                        pbar.update(1)

                        if all(dones.values()):
                            self._log_episode_metrics(total_steps)
                            _ep_task = int(self._episode_task_count)
                            _ep_conflict = float(self._episode_conflict_sum) / max(1, self._episode_steps)
                            self._reset_episode_accumulators()
                            pbar.set_postfix_str(
                                f"ep={int(episode)} tc={_ep_task} cr={_ep_conflict:.3f}",
                                refresh=False,
                            )
                            episode += 1
                            obs = None
                            info = None
                        else:
                            obs = next_obs
                            info = next_info

                    if hasattr(self.agent, "set_training_mode"):
                        self.agent.set_training_mode(True)

                    if horizon_steps == 0:
                        break

                    # Finalize rollout: shift values to get next_values, discard last step
                    if hasattr(self.agent, "buffer") and hasattr(self.agent.buffer, "finalize_rollout"):
                        self.agent.buffer.finalize_rollout()

                    self._check_stop()
                    lr_now = self._compute_scheduled_lr(int(total_steps), int(target_total_steps))
                    if lr_now is not None:
                        self._set_optimizer_lr(float(lr_now))
                    if hasattr(self.agent, "set_training_step"):
                        self.agent.set_training_step(int(total_steps))
                    while True:
                        try:
                            self.agent.update()
                            break
                        except Exception as exc:
                            if not self._looks_like_oom(exc):
                                raise
                            self._clear_memory()
                            raise RuntimeError(
                                "CUDA out of memory during the main training phase. "
                                "Try reducing batch_size or the model size."
                            ) from exc
                    self._last_safe_episode = int(episode)
                    self._last_safe_total_steps = int(total_steps)
                    next_save_step = self._maybe_save(
                        int(total_steps),
                        next_save_step,
                        int(save_interval_steps),
                        int(episode),
                        int(env_seed),
                        int(horizon_len),
                        int(max_train_steps),
                        int(target_total_steps),
                    )

            self._save_checkpoint(
                total_steps=int(total_steps),
                episode=int(episode),
                env_seed=int(env_seed),
                horizon_len=int(horizon_len),
                max_train_steps=int(max_train_steps),
                target_total_steps=int(target_total_steps),
            )
        except KeyboardInterrupt:
            ckpt_path = self._save_interrupt_checkpoint(
                total_steps=int(total_steps),
                episode=int(episode),
                env_seed=int(env_seed),
                horizon_len=int(horizon_len),
                max_train_steps=int(max_train_steps),
                target_total_steps=int(target_total_steps),
            )
            raise
        finally:
            self.close()
            if self.logger is not None:
                self.logger.close()
def run_training(run_cfg: RunConfig, *, checkpoint_path: str = "",
                 start_episode: int = 0, total_steps: int = 0) -> None:
    trainer = Trainer(run_cfg.env, run_cfg.algo)

    # Persist the resolved run config for reproducibility and evaluation reuse
    run_dir = getattr(run_cfg.algo, "model_dir", None)
    root_dir = os.path.dirname(str(run_dir)) if run_dir else os.getcwd()
    config_dir = os.path.join(root_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    run_config_path = os.path.join(config_dir, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_cfg.to_dict(), f, ensure_ascii=False, indent=2)

    start_episode = max(0, int(start_episode))
    total_steps = max(0, int(total_steps))

    checkpoint_path = str(checkpoint_path or "").strip()
    if checkpoint_path != "":
        spec = load_training_checkpoint(
            trainer.agent,
            checkpoint_path,
            expected_algorithm=trainer._algo_name(),
        )
        resume_meta = dict(getattr(spec, "checkpoint_meta", {}) or {})
        start_episode = int(resume_meta.get("episode", start_episode) or start_episode)
        total_steps = int(resume_meta.get("total_steps", total_steps) or total_steps)

    trainer.run(start_episode=start_episode, total_steps=total_steps)


def main(argv: Optional[List[str]] = None) -> int:
    run_cfg, meta = load_and_build_config("train", argv=argv)
    checkpoint_path = str(meta.get("checkpoint_path", "") or "").strip()
    extra = {
        "start_episode": max(0, int(meta.get("start_episode", 0) or 0)),
        "total_steps": max(0, int(meta.get("total_steps", 0) or 0)),
        "checkpoint_path": checkpoint_path,
    }
    if checkpoint_path != "":
        extra["resume_note"] = "After loading a checkpoint, the episode / total_steps recorded inside it take precedence."
    if not confirm_resolved_config("train", run_cfg, extra=extra):
        return 1
    run_training(
        run_cfg,
        checkpoint_path=checkpoint_path,
        start_episode=int(meta.get("start_episode", 0) or 0),
        total_steps=int(meta.get("total_steps", 0) or 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
