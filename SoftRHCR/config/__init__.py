from SoftRHCR.config.configBase import (
    CommonConfig,
    MAPPOConfig, IPPOConfig,
    SoftRHCRConfig, SoftRHCRMAPPOConfig,
    GateBlendConfig, GateBlendMAPPOConfig,
    FollowPlannerConfig,
    EnvConfig, PlannerConfig, RunConfig,
)
from SoftRHCR.config.config_loader import load_and_build_config

all_configs = [
    CommonConfig,
    EnvConfig, PlannerConfig, RunConfig,
    MAPPOConfig, IPPOConfig,
    SoftRHCRConfig, SoftRHCRMAPPOConfig,
    GateBlendConfig, GateBlendMAPPOConfig,
    FollowPlannerConfig,
]
