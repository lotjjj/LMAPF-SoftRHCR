import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

SPATIAL_BACKBONES: Tuple[str, ...] = ("default",)


def orthogonal_init_(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

def _make_norm2d(norm_type: str, num_channels: int) -> nn.Module:
    t = (norm_type or "bn").strip().lower()
    if t in ("bn", "batchnorm", "batch_norm"):
        return nn.BatchNorm2d(num_channels)
    if t in ("gn", "groupnorm", "group_norm"):
        groups = 8
        while groups > 1 and (num_channels % groups) != 0:
            groups //= 2
        return nn.GroupNorm(num_groups=max(1, groups), num_channels=num_channels)
    if t in ("none", "no", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type={norm_type}")


def available_backbones() -> Tuple[str, ...]:
    return SPATIAL_BACKBONES


def _normalize_spatial_backbone(backbone: str) -> str:
    bb = (backbone or "default").strip().lower()
    if bb in ("default", "cnn", "basic"):
        return "default"
    raise ValueError(f"Unsupported backbone={backbone}")


def _build_spatial_cnn(backbone: str, in_channels: int, norm_type: str) -> nn.Module:
    bb = _normalize_spatial_backbone(backbone)
    if bb == "default":
        return _CNNDefault(in_channels, norm_type=norm_type)
    raise ValueError(f"Unsupported backbone={backbone}")


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, norm_type: str = "bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = _make_norm2d(norm_type, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _make_norm2d(norm_type, channels)

        orthogonal_init_(self.conv1, gain=nn.init.calculate_gain("relu"))
        orthogonal_init_(self.bn1)
        orthogonal_init_(self.conv2, gain=nn.init.calculate_gain("relu"))
        orthogonal_init_(self.bn2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class _CNNDefault(nn.Module):
    def __init__(self, in_channels: int, norm_type: str = "bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = _make_norm2d(norm_type, 32)
        self.res1 = ResidualConvBlock(32, norm_type=norm_type)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = _make_norm2d(norm_type, 64)
        self.res2 = ResidualConvBlock(64, norm_type=norm_type)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res2(x)
        x = self.flatten(self.pool(x))
        return x


class FeatureExtractor(nn.Module):
    """CNN + MLP hybrid feature extractor."""
    def __init__(
        self,
        fov_shape=(5, 11, 11),
        msg_shape=(10, 11, 11),
        self_state_dim=25,
        feature_dim=128,
        norm_type: str = "gn",
        backbone: str = "default",
        norm_after_concat: str = "none",
        msgs_mode: str = "single",
    ):
        super(FeatureExtractor, self).__init__()
        fov_channels, h, w = fov_shape
        msg_channels = msg_shape[0]

        spatial_backbone = _normalize_spatial_backbone(backbone)
        self._msgs_mode = str(msgs_mode or "single").strip().lower()
        if self._msgs_mode not in ("single", "dual"):
            self._msgs_mode = "single"
        in_channels = fov_channels + msg_channels
        self.cnn = _build_spatial_cnn(spatial_backbone, in_channels, norm_type)
        
        self.mlp = nn.Sequential(
            nn.Linear(self_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            cnn_out = self.cnn(dummy)
            cnn_out_dim = cnn_out.shape[1]
        concat_dim = int(cnn_out_dim + 64)
        nac = (norm_after_concat or "none").strip().lower()
        if nac in ("ln", "layernorm", "layer_norm"):
            self.norm_after_concat = nn.LayerNorm(concat_dim)
        elif nac in ("none", "no", "identity"):
            self.norm_after_concat = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_after_concat={norm_after_concat}")
        self.fc_out = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        self.apply(lambda m: orthogonal_init_(m, gain=nn.init.calculate_gain("relu")))

    def forward(self, obs_dict):
        """Args:
        obs_dict: a dict or Batch containing 'fov', 'self_states' and 'msgs'.
        """
        fov = obs_dict['fov']
        msgs = obs_dict['msgs']
        self_states = obs_dict['self_states']

        bsz = int(fov.shape[0])
        fov = fov.reshape(bsz, *fov.shape[1:])
        msgs = msgs.reshape(bsz, *msgs.shape[1:])
        self_states = self_states.reshape(bsz, self_states.shape[-1])

        cnn_in = torch.cat([fov, msgs], dim=1)
        cnn_out = self.cnn(cnn_in)
        mlp_out = self.mlp(self_states)

        combined = torch.cat([cnn_out, mlp_out], dim=1)
        combined = self.norm_after_concat(combined)
        return self.fc_out(combined)

class Actor(nn.Module):
    """Policy network (actor)."""
    def __init__(self, feature_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        orthogonal_init_(self.fc1, gain=nn.init.calculate_gain("relu"))
        orthogonal_init_(self.fc2, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Critic(nn.Module):
    """State value network (V) or Q-network (critic)."""
    def __init__(self, feature_dim, out_dim=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)
        orthogonal_init_(self.fc1, gain=nn.init.calculate_gain("relu"))
        orthogonal_init_(self.fc2, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MAPPOCritic(nn.Module):
    """MAPPO global-state critic — consumes features from the shared FeatureExtractor.

    Input: concatenated features of all agents, shape (B, N, feature_dim) or (N, feature_dim).
    - per_agent mode: outputs (B, n_agents) — one value estimate per agent.
    - team mode:      outputs (B, 1) — a single team value.
    """
    def __init__(self, feature_dim: int, n_agents: int, mode: str = "per_agent"):
        super(MAPPOCritic, self).__init__()
        self.mode = mode
        self.n_agents = n_agents
        input_dim = feature_dim * n_agents
        out_dim = n_agents if mode == "per_agent" else 1
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_dim)
        orthogonal_init_(self.fc1, gain=nn.init.calculate_gain("relu"))
        orthogonal_init_(self.fc2, gain=nn.init.calculate_gain("relu"))
        orthogonal_init_(self.fc3, gain=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            features = features.unsqueeze(0)
        flat = features.reshape(features.shape[0], -1)
        x = F.relu(self.fc1(flat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)