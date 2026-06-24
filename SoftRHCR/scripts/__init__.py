from importlib import import_module
from typing import Any

__all__ = [
    "Trainer",
    "set_global_seeds",
    "run_training",
    "build_cfg_from_cli",
    "build_experiment",
    "Experiment",
    "evaluate_model",
    "evaluate_planner",
    "evaluate_agent_rollout",
    "EvaluationStopped",
    "extract_obs_info",
    "strip_ansi",
    "looks_like_progress_line",
    "infer_step_from_name",
    "create_agent",
    "load_agent_for_evaluation",
    "get_action_masks",
    "select_actions",
    "build_eval_seed_list",
    "split_eval_seeds_evenly",
    "format_seed_group_compact",
    "summarize_seed_groups",
    "env_cfg_to_dict",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is not None:
        module = import_module(module_name)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_EXPORT_MODULES = {
    "Trainer": "SoftRHCR.scripts.trainer",
    "set_global_seeds": "SoftRHCR.scripts.trainer",
    "run_training": "SoftRHCR.scripts.trainer",
    "build_cfg_from_cli": "SoftRHCR.scripts.runtime_experiment",
    "build_experiment": "SoftRHCR.scripts.runtime_experiment",
    "Experiment": "SoftRHCR.scripts.runtime_experiment",
    "evaluate_model": "SoftRHCR.scripts.evaluation",
    "evaluate_planner": "SoftRHCR.scripts.evaluation",
    "evaluate_agent_rollout": "SoftRHCR.scripts.evaluation",
    "EvaluationStopped": "SoftRHCR.scripts.evaluation",
    "extract_obs_info": "SoftRHCR.scripts.evaluation",
    "strip_ansi": "SoftRHCR.scripts.evaluation",
    "looks_like_progress_line": "SoftRHCR.scripts.evaluation",
    "infer_step_from_name": "SoftRHCR.scripts.evaluation",
    "create_agent": "SoftRHCR.scripts.evaluation",
    "load_agent_for_evaluation": "SoftRHCR.scripts.evaluation",
    "get_action_masks": "SoftRHCR.scripts.evaluation",
    "select_actions": "SoftRHCR.scripts.evaluation",
    "build_eval_seed_list": "SoftRHCR.scripts.evaluation",
    "split_eval_seeds_evenly": "SoftRHCR.scripts.evaluation",
    "format_seed_group_compact": "SoftRHCR.scripts.evaluation",
    "summarize_seed_groups": "SoftRHCR.scripts.evaluation",
    "env_cfg_to_dict": "SoftRHCR.scripts.evaluation",
}
