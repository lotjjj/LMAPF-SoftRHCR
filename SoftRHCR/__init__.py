from importlib import import_module

from SoftRHCR.config.configBase import (
    EnvConfig, EnvParams, PlannerConfig, RunConfig,
    IPPOConfig, MAPPOConfig,
    SoftRHCRConfig, SoftRHCRMAPPOConfig, GateBlendConfig, GateBlendMAPPOConfig,
    FollowPlannerConfig, CommonConfig, algo_config_from_dict,
)
from SoftRHCR.config.config_loader import load_and_build_config, USERCFG_DIR
from SoftRHCR.config.registry import (
    create_agent, algorithm_options, algorithm_registry_entries,
    get_algorithm_entry, normalize_algorithm_token, algorithm_name_from_agent,
    agent_select_action_requires_info,
)
from SoftRHCR.modules.device import resolve_device
from SoftRHCR.modules.model_io import (
    create_and_load_agent_from_model_spec, resolve_model_load_spec, load_training_checkpoint,
    ModelLoadSpec,
)
from SoftRHCR.modules.network import available_backbones
from SoftRHCR.config.module_catalog import extractor_options
from SoftRHCR.config.planner_catalog import (
    planner_options, planner_field_specs,
    normalize_planner_type_token, default_planner_overrides,
)

__all__ = [
    "EnvConfig", "EnvParams", "PlannerConfig", "RunConfig",
    "IPPOConfig", "MAPPOConfig",
    "SoftRHCRConfig", "SoftRHCRMAPPOConfig", "GateBlendConfig", "GateBlendMAPPOConfig",
    "FollowPlannerConfig", "CommonConfig", "algo_config_from_dict",
    "load_and_build_config", "USERCFG_DIR",
    "create_agent", "algorithm_options", "algorithm_registry_entries",
    "get_algorithm_entry", "normalize_algorithm_token", "algorithm_name_from_agent",
    "agent_select_action_requires_info",
    "resolve_device",
    "create_and_load_agent_from_model_spec", "resolve_model_load_spec", "load_training_checkpoint", "ModelLoadSpec",
    "available_backbones",
    "Trainer",
    "evaluate_model", "evaluate_planner", "evaluate_agent_rollout",
    "EvaluationStopped",
    "extract_obs_info", "env_cfg_to_dict",
    "extractor_options",
    "planner_options", "planner_field_specs",
    "normalize_planner_type_token", "default_planner_overrides",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)


_LAZY_EXPORT_MODULES = {
    "Trainer": "SoftRHCR.scripts.trainer",
    "evaluate_model": "SoftRHCR.scripts.evaluation",
    "evaluate_planner": "SoftRHCR.scripts.evaluation",
    "evaluate_agent_rollout": "SoftRHCR.scripts.evaluation",
    "EvaluationStopped": "SoftRHCR.scripts.evaluation",
    "extract_obs_info": "SoftRHCR.scripts.evaluation",
    "env_cfg_to_dict": "SoftRHCR.scripts.evaluation",
}
