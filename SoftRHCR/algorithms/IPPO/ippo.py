import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from SoftRHCR.algorithms.agentBase import AgentBase
from SoftRHCR.algorithms.PPO.ppo_update import ippo_update
from SoftRHCR.modules.network import FeatureExtractor, Actor, Critic
from SoftRHCR.modules.replay_buffer import PPOBuffer, _flatten_self_states
from SoftRHCR.modules.device import move_optimizer_state_to_device, resolve_device, try_load_optimizer_state
from typing import Dict, Any, List, Optional

class IPPOAgent(AgentBase):
    def __init__(self, config, obs_info, action_dim, n_agents, n_actions=None):
        super().__init__()
        self.device = resolve_device(getattr(config, "device", "auto"))
        self.n_agents = n_agents
        self.action_dim = action_dim if n_actions is None else n_actions
        self.n_actions = self.action_dim
        self.gamma = getattr(config, "gamma", 0.99)
        self.lr = getattr(config, "lr", 3e-4)
        self.ppo_epoch = getattr(config, "ppo_epoch", 4)
        self.clip_param = getattr(config, "clip_param", 0.2)
        self.entropy_coef = getattr(config, "entropy_coef", 0.01)
        self.value_loss_coef = getattr(config, "value_loss_coef", 0.5)
        self.use_gae = getattr(config, "use_gae", True)
        self.gae_lambda = getattr(config, "gae_lambda", 0.95)
        self.horizon_len = getattr(config, "horizon_len", 400)
        self.batch_size = getattr(config, "batch_size", 0)
        self.grad_clip_max_norm = getattr(config, "grad_clip_max_norm", None)
        self.weight_decay = float(getattr(config, "weight_decay", 1e-4))
        self._optimizer_cls = self._resolve_optimizer_cls(config)
        self._training_steps = 0
        self.reward_mode = getattr(config, "reward_mode", "aggressive")
        self.critic_mode = getattr(config, "critic_mode", "homogeneous")

        fov_shape = tuple(obs_info.get("fov", (5, 11, 11)))
        msg_shape = tuple(obs_info.get("msgs", (10, 11, 11)))
        self.obs_shape = {
            "fov": fov_shape,
            "msgs": msg_shape,
            "self_states": int(obs_info.get("self_states", 25)),
        }
        self.map_width = int(obs_info.get("map_width", 1))
        self.map_height = int(obs_info.get("map_height", 1))
        self.k_horizon = int(msg_shape[0]) if len(msg_shape) > 0 else 0
        self.feature_dim = getattr(config, "feature_dim", 128)
        self._last_adapted_obs: Optional[Dict[str, Any]] = None

        self.extractor = FeatureExtractor(
            fov_shape=fov_shape,
            msg_shape=msg_shape,
            self_state_dim=int(obs_info.get("self_states", 25)),
            feature_dim=self.feature_dim,
            norm_type=getattr(config, "norm_type", "gn"),
            backbone=getattr(config, "extractor_backbone", "default"),
            norm_after_concat=getattr(config, "norm_after_concat", "none"),
        ).to(self.device)
        self.actor = Actor(self.feature_dim, self.n_actions).to(self.device)
        self.critic = Critic(self.feature_dim, 1).to(self.device)

        self.params = list(self.extractor.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = self._optimizer_cls(self.params, lr=self.lr, weight_decay=self.weight_decay)
        self.buffer = PPOBuffer(
            n_steps=self.horizon_len,
            n_agents=n_agents,
            obs_shape=obs_info,
            action_dim=self.n_actions,
            device=self.device,
        )

    @staticmethod
    def _resolve_optimizer_cls(config) -> type:
        name = str(getattr(config, "optimizer", "AdamW")).strip()
        cls = getattr(optim, name, None)
        if cls is None or not isinstance(cls, type):
            raise ValueError(f"Unsupported optimizer: {name!r} (expected: Adam, AdamW)")
        return cls

    @staticmethod
    def _agent_ids(obs: Dict[str, Any]) -> List[str]:
        return sorted([str(aid) for aid in obs.keys()], key=lambda x: int(str(x).split("_")[-1]))

    def adapt_observations(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        adapted: Dict[str, Any] = {}
        msg_shape = tuple(self.obs_shape["msgs"])
        zero_msgs = np.zeros(msg_shape, dtype=np.float32)
        for aid, agent_obs in (obs or {}).items():
            if not isinstance(agent_obs, dict):
                continue
            adapted[str(aid)] = {
                "fov": np.asarray(agent_obs.get("fov", np.zeros(self.obs_shape["fov"], dtype=np.float32)), dtype=np.float32),
                "msgs": np.asarray(agent_obs.get("msgs", zero_msgs), dtype=np.float32),
                "self_states": agent_obs.get("self_states", {}),
            }
        self._last_adapted_obs = adapted
        return adapted

    def get_last_adapted_observations(self) -> Optional[Dict[str, Any]]:
        return self._last_adapted_obs

    def get_observation_aux(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def _stack_obs(self, obs: Dict[str, Any], agent_ids: List[str]) -> Dict[str, torch.Tensor]:
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
                np.stack([_flatten_self_states(obs[aid]["self_states"]) for aid in agent_ids], axis=0),
                device=self.device,
            ),
        }

    def _stack_action_masks(self, action_masks: Optional[Dict[str, Any]], agent_ids: List[str]) -> Optional[torch.Tensor]:
        if not isinstance(action_masks, dict):
            return None
        return torch.as_tensor(
            np.stack(
                [
                    np.asarray(
                        action_masks.get(aid, np.ones((self.n_actions,), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    for aid in agent_ids
                ],
                axis=0,
            ),
            device=self.device,
        )

    def select_action(self, obs, evaluation=False, action_masks=None, **kwargs):
        policy_obs = self.adapt_observations(obs)
        agent_ids = self._agent_ids(policy_obs)
        obs_tensor = self._stack_obs(policy_obs, agent_ids)
        action_mask_tensor = self._stack_action_masks(action_masks, agent_ids)

        with torch.no_grad():
            features = self.extractor(obs_tensor)
            logits = self.actor(features)
            if action_mask_tensor is not None:
                logits = logits.masked_fill(action_mask_tensor <= 0, -1e9)
            dist = Categorical(logits=logits)
            actions_tensor = torch.argmax(logits, dim=-1) if evaluation else dist.sample()
            logprobs_tensor = dist.log_prob(actions_tensor)
            values_tensor = self.critic(features).squeeze(-1)

        actions = {aid: int(actions_tensor[idx].detach().cpu().item()) for idx, aid in enumerate(agent_ids)}
        logprobs = {aid: float(logprobs_tensor[idx].detach().cpu().item()) for idx, aid in enumerate(agent_ids)}
        values = {aid: float(values_tensor[idx].detach().cpu().item()) for idx, aid in enumerate(agent_ids)}
        return actions, logprobs, values

    def get_actions(self, obs, action_masks=None, **kwargs):
        actions, _, _ = self.select_action(
            obs,
            evaluation=bool(kwargs.get("evaluation", False)),
            action_masks=action_masks,
        )
        return actions

    def evaluate_actions(self, obs, actions, action_masks=None):
        features = self.extractor(obs)
        logits = self.actor(features)
        if action_masks is not None:
            logits[action_masks == 0] = -1e9
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy().mean()
        values = self.critic(features).squeeze(-1)
        return values, log_probs, entropy

    def get_value(self, obs):
        return self.get_values(obs)

    def get_values(self, obs):
        policy_obs = self.adapt_observations(obs)
        agent_ids = self._agent_ids(policy_obs)
        obs_tensor = self._stack_obs(policy_obs, agent_ids)
        with torch.no_grad():
            features = self.extractor(obs_tensor)
            values = self.critic(features).squeeze(-1)
        return {aid: float(values[idx].detach().cpu().item()) for idx, aid in enumerate(agent_ids)}

    def compute_returns(self, rewards, values, dones, next_values):
        if self.use_gae:
            gae = 0
            returns = torch.zeros_like(rewards)
            for t in reversed(range(rewards.shape[0])):
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t].float()) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t].float()) * gae
                returns[t] = gae + values[t]
        else:
            returns = torch.zeros_like(rewards)
            running_return = next_values[-1]
            for t in reversed(range(rewards.shape[0])):
                running_return = rewards[t] + self.gamma * running_return * (1 - dones[t].float())
                returns[t] = running_return
        return returns

    def update(self):
        return ippo_update(self)

    def compute_intrinsic_rewards(self, *args, **kwargs) -> Dict[str, float]:
        # IPPO does not implement intrinsic rewards by default; users may override
        # this, but it will not affect SoftRHCR through the MRO (MRO isolation).
        return {}

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

    def load_model(self, path, load_critic=True, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.extractor.load_state_dict(checkpoint["extractor"])
        self.actor.load_state_dict(checkpoint["actor"])
        if load_critic and "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer and "optimizer" in checkpoint:
            try_load_optimizer_state(self.optimizer, checkpoint["optimizer"], self.device)
        self._training_steps = int(checkpoint.get("training_steps", 0))

    def set_training_mode(self, mode=True):
        self.extractor.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)

    def set_training_step(self, step):
        self._training_steps = int(step)

    def reset_episode(self):
        self._last_adapted_obs = None

    def reset_done_agents(self, dones):
        return None

    def to_device(self, device):
        self.device = resolve_device(device)
        self.extractor = self.extractor.to(self.device)
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.buffer.device = self.device
        move_optimizer_state_to_device(self.optimizer, self.device)
