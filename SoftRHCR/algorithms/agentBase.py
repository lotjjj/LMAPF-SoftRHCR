import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Union
from SoftRHCR.modules.device import resolve_device
from SoftRHCR.modules.observation_adapter import DefaultObservationAdapter, ObservationAdapterBase


class AgentBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True
        self.device = resolve_device("auto")

    def get_actions(self, obs, action_masks=None, **kwargs):
        raise NotImplementedError

    def get_value(self, obs):
        """Return per-agent value estimates."""
        return None

    def save_model(self, path):
        torch.save({
            "model_state_dict": self.state_dict(),
        }, path)

    def load_model(self, path, load_critic=True, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])

    def set_training_mode(self, mode=True):
        self.training = mode
        nn.Module.train(self, mode)

    def set_device(self, device):
        self.device = resolve_device(device)
        self.to(self.device)

    def setup_observation_adapter(self, obs_info: Dict[str, Any], n_agents: int, config: Any = None) -> ObservationAdapterBase:
        return DefaultObservationAdapter(obs_info, n_agents, config=config)

    def compute_intrinsic_rewards(self, *args, **kwargs) -> Dict[str, float]:
        """Compute intrinsic rewards; the default implementation returns none.

        Algorithms that need intrinsic rewards (e.g. SoftRHCR) should override
        this method in their own class or a mixin.
        """
        return {}

    @staticmethod
    def _to_scalar(v):
        if v is None:
            return None
        if isinstance(v, (bool, int, float)):
            return float(v)
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.detach().cpu().item())
            return None
        try:
            import numpy as np
            if isinstance(v, np.generic):
                return float(v)
            if isinstance(v, np.ndarray) and v.size == 1:
                return float(v.reshape(()))
        except Exception:
            pass
        return None
