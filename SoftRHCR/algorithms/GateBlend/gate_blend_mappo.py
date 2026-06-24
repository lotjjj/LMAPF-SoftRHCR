from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from SoftRHCR.algorithms.MAPPO.mappo import MAPPOAgent
from SoftRHCR.modules.device import move_optimizer_state_to_device, try_load_optimizer_state


class GateBlendMAPPOAgent(MAPPOAgent):
    def __init__(self, config, obs_info, action_dim, n_agents, n_actions=None):
        super().__init__(config, obs_info, action_dim, n_agents, n_actions=n_actions)

        self.reward_mode = getattr(config, "reward_mode", "legacy")
        self.lambda1 = getattr(config, "lambda1", 0.1)
        self.lambda2 = getattr(config, "lambda2", 0.1)
        self.lambda3 = getattr(config, "lambda3", 0.02)
        self.gate_no_ppo_grad = getattr(config, "gate_no_ppo_grad", False)
        self.specialized_lambda_anneal = getattr(config, "specialized_lambda_anneal", True)
        self.specialized_lambda_min = getattr(config, "specialized_lambda_min", 0.0)
        self.bce_lambda_anneal = getattr(config, "bce_lambda_anneal", True)
        self.bce_lambda_min = getattr(config, "bce_lambda_min", 0.0)
        self.bce_lambda_exp_k = getattr(config, "bce_lambda_exp_k", 10.0)
        self.specialized_huber_delta = getattr(config, "specialized_huber_delta", 1.0)
        self.gate_hidden_dim = getattr(config, "gate_hidden_dim", 128)
        self.hard_tau = getattr(config, "hard_tau", 0.8)
        self.fp_soft_beta = getattr(config, "fp_soft_beta", 2.0)
        self.kpc_horizon = getattr(config, "kpc_horizon", 10)
        self.kpc_exp_beta = getattr(config, "kpc_exp_beta", 10.0)
        self.conflict_sparse_gate_supervision = getattr(config, "conflict_sparse_gate_supervision", True)

        self.gate_net = nn.Sequential(
            nn.Linear(self.feature_dim, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.gate_hidden_dim, 1),
        ).to(self.device)

        self.gate_optimizer = self._optimizer_cls(
            list(self.gate_net.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self._training_step = 0

    def get_actions(self, obs, action_masks=None, **kwargs):
        fp_actions = kwargs.get("fp_actions") if kwargs else None
        use_fp = kwargs.get("use_fp") if kwargs else None

        if fp_actions is not None and use_fp is not None:
            return self._select_action_with_gate(obs, action_masks, fp_actions, use_fp)

        return super().get_actions(obs, action_masks=action_masks, **kwargs)

    def _select_action_with_gate(self, obs, action_masks, fp_actions, use_fp):
        with torch.no_grad():
            features = self.extractor(obs)
            logits = self.actor(features)
            if action_masks is not None:
                logits[action_masks == 0] = -1e9
            dist = Categorical(logits=logits)
            rl_actions = dist.sample()
            gate_logit = self.gate_net(features).squeeze(-1)
            gate_prob = torch.sigmoid(gate_logit)

        actions = {}
        n_agents = len(obs["fov"])
        for i in range(n_agents):
            agent_id = f"agv_{i}"
            if fp_actions is not None and agent_id in fp_actions and use_fp is not None and agent_id in use_fp:
                if bool(use_fp[agent_id]):
                    p = float(gate_prob[i].cpu().item())
                    if np.random.random() < p:
                        actions[agent_id] = int(rl_actions[i].cpu().item())
                    else:
                        actions[agent_id] = int(fp_actions[agent_id])
                else:
                    actions[agent_id] = int(rl_actions[i].cpu().item())
            else:
                actions[agent_id] = int(rl_actions[i].cpu().item())
        return actions

    def update(self):
        # Cache buffer data before PPO update (which clears the buffer)
        has_data = int(self.buffer.ptr) > 0
        if has_data:
            cached_obs = {
                k: v.clone() for k, v in
                {kk: vv.reshape(vv.shape[0] * vv.shape[1], *vv.shape[2:])
                 for kk, vv in self.buffer.get_all()["obs"].items()}.items()
            }

        # MAPPO update via parent
        metrics = super().update()

        # Gate network training: entropy regularisation only
        if has_data:
            with torch.no_grad():
                features = self.extractor(cached_obs)
            gate_logit = self.gate_net(features).squeeze(-1)
            gate_prob = torch.sigmoid(gate_logit)

            gate_entropy = -(
                gate_prob * (gate_prob + 1e-8).log()
                + (1 - gate_prob) * (1 - gate_prob + 1e-8).log()
            ).mean()
            gate_loss = -0.01 * gate_entropy

            self.gate_optimizer.zero_grad()
            gate_loss.backward()
            self.gate_optimizer.step()

        return metrics

    def save_model(self, path, meta=None):
        payload = {
            "extractor": self.extractor.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "gate_net": self.gate_net.state_dict(),
            "gate_optimizer": self.gate_optimizer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
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
        if "gate_net" in checkpoint:
            self.gate_net.load_state_dict(checkpoint["gate_net"])
        if load_optimizer and "optimizer" in checkpoint:
            try_load_optimizer_state(self.optimizer, checkpoint["optimizer"], self.device)
            try_load_optimizer_state(self.gate_optimizer, checkpoint.get("gate_optimizer"), self.device)

    def set_training_mode(self, mode=True):
        super().set_training_mode(mode)
        self.gate_net.train(mode)

    def to_device(self, device):
        super().to_device(device)
        self.gate_net = self.gate_net.to(self.device)
        move_optimizer_state_to_device(self.gate_optimizer, self.device)
