from typing import Any, Dict, Optional

import torch

from SoftRHCR.algorithms.agentBase import AgentBase
from SoftRHCR.modules.device import resolve_device


class FollowPlannerAgent(AgentBase):
    def __init__(self, config, obs_info, action_dim, n_agents, n_actions=None):
        super().__init__()
        self.device = resolve_device(getattr(config, "device", "auto"))
        self.n_agents = n_agents
        self.action_dim = action_dim if n_actions is None else n_actions
        self.n_actions = self.action_dim

    def select_action(self, obs, action_masks=None, **kwargs):
        return self.get_actions(obs, action_masks=action_masks, **kwargs)

    def get_actions(self, obs, action_masks=None, **kwargs):
        actions = {}
        if kwargs is not None and "fp_actions" in kwargs and kwargs["fp_actions"] is not None:
            fp_actions = kwargs["fp_actions"]
            if isinstance(fp_actions, dict):
                for agent_id in obs.keys() if isinstance(obs, dict) else []:
                    if agent_id in fp_actions:
                        actions[agent_id] = int(fp_actions[agent_id])
                    else:
                        actions[agent_id] = 0
        else:
            if isinstance(obs, dict):
                for agent_id in obs.keys():
                    actions[agent_id] = 0
        return actions

    def get_value(self, obs):
        return None

    def save_model(self, path):
        pass

    def load_model(self, path, load_critic=True, load_optimizer=True):
        pass

    def set_training_mode(self, mode=True):
        pass

    def to_device(self, device):
        self.device = resolve_device(device)
