from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Optional

from SoftRHCR.config.configBase import RunConfig

_SECTION_WIDTH = 96
_KEY_WIDTH = 24


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _prune_payload(value: Any) -> Any:
    normalized = _jsonable(value)
    if isinstance(normalized, dict):
        result: dict[str, Any] = {}
        for key, item in normalized.items():
            pruned = _prune_payload(item)
            if _is_empty(pruned):
                continue
            result[str(key)] = pruned
        return result
    if isinstance(normalized, list):
        result_list = []
        for item in normalized:
            pruned = _prune_payload(item)
            if _is_empty(pruned):
                continue
            result_list.append(pruned)
        return result_list
    return normalized


def _ordered_mapping(
    payload: Any,
    preferred_keys: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    pruned = _prune_payload(payload)
    if not isinstance(pruned, dict):
        return {}
    result: dict[str, Any] = {}
    seen: set[str] = set()
    for key in preferred_keys or ():
        if key in pruned:
            result[key] = pruned[key]
            seen.add(key)
    for key in sorted(pruned.keys()):
        if key in seen:
            continue
        result[key] = pruned[key]
    return result


def _inline_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _render_nested_value(value: Any, indent: int) -> list[str]:
    rendered = json.dumps(value, ensure_ascii=False, indent=2).splitlines()
    pad = " " * indent
    return [f"{pad}{line}" for line in rendered]


def _section_lines(
    title: str,
    payload: Any,
    preferred_keys: Optional[Iterable[str]] = None,
) -> list[str]:
    mapping = _ordered_mapping(payload, preferred_keys=preferred_keys)
    if len(mapping) == 0:
        return []
    lines = [f"[{title}]"]
    for key, value in mapping.items():
        label = f"  {str(key):<{_KEY_WIDTH}} : "
        if isinstance(value, (dict, list)):
            lines.append(f"{label.rstrip()}")
            lines.extend(_render_nested_value(value, indent=len(label)))
            continue
        lines.append(f"{label}{_inline_value(value)}")
    lines.append("")
    return lines


def print_resolved_config_summary(
    mode: str,
    run_cfg: RunConfig,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    mode_key = str(mode or "").strip().lower()
    env_mode = {
        "train": "train",
        "evaluate": "eval",
        "experiment": "runtime",
    }.get(mode_key, mode_key)
    mode_label = {
        "train": "Train",
        "evaluate": "Evaluate",
        "experiment": "Runtime",
    }.get(mode_key, mode_key or "Run")

    env_shared_payload = {
        "fov_size": int(run_cfg.env.fov_size),
        "kstep_conflict_check": int(run_cfg.env.kstep_conflict_check),
        "targets_on_shelf": bool(run_cfg.env.targets_on_shelf),
        "eval_episodes": int(run_cfg.env.eval_episodes),
    }
    env_mode_payload = getattr(run_cfg.env, env_mode)
    planner_payload = {
        "planner_type": str(run_cfg.env.planner.planner_type),
        "planner_overrides": dict(run_cfg.env.planner.overrides),
        "resolved_planner_args": run_cfg.env.planner.resolved_planner_args(),
    }
    overview_payload = {
        "mode": mode_key,
        "algorithm": str(run_cfg.algorithm),
        "device": str(getattr(run_cfg.algo, "device", "") or ""),
        "model_dir": str(getattr(run_cfg.algo, "model_dir", "") or ""),
    }
    algo_preferred_keys = (
        "seed",
        "device",
        "horizon_len",
        "max_train_steps",
        "save_interval_steps",
        "buffer_size",
        "lr",
        "min_lr",
        "lr_schedule",
        "ppo_epoch",
        "clip_param",
        "entropy_coef",
        "value_loss_coef",
        "gamma",
        "gae_lambda",
    )
    extra_preferred_keys = (
        "execution_mode",
        "runtime_mode",
        "start_episode",
        "total_steps",
        "checkpoint_path",
        "model_path",
        "run_config_path",
        "eval_threads",
        "task_limit",
        "msgs_mode",
        "resume_note",
    )

    lines = [
        "",
        "=" * _SECTION_WIDTH,
        f"{mode_label} configuration review",
        "=" * _SECTION_WIDTH,
        "",
    ]
    lines.extend(
        _section_lines(
            "Overview",
            overview_payload,
            preferred_keys=("mode", "algorithm", "device", "model_dir"),
        )
    )
    lines.extend(
        _section_lines(
            "Env (shared)",
            env_shared_payload,
            preferred_keys=("fov_size", "kstep_conflict_check", "targets_on_shelf", "eval_episodes"),
        )
    )
    lines.extend(
        _section_lines(
            f"Env ({env_mode})",
            env_mode_payload,
            preferred_keys=("num_agvs", "map_size", "max_episode_steps", "seed", "render_mode", "render_interval_s"),
        )
    )
    lines.extend(
        _section_lines(
            "Planner",
            planner_payload,
            preferred_keys=("planner_type", "resolved_planner_args", "planner_overrides"),
        )
    )
    lines.extend(_section_lines("Algorithm", run_cfg.algo, preferred_keys=algo_preferred_keys))
    lines.extend(_section_lines("Extra", extra or {}, preferred_keys=extra_preferred_keys))
    lines.extend(
        [
            "Please review the resolved configuration above.",
            "Enter y or yes to confirm and start; any other input will cancel this run.",
            "",
        ]
    )
    print("\n".join(lines), flush=True)


def confirm_resolved_config(
    mode: str,
    run_cfg: RunConfig,
    extra: Optional[dict[str, Any]] = None,
) -> bool:
    print_resolved_config_summary(mode=mode, run_cfg=run_cfg, extra=extra)
    try:
        answer = input("Confirm and start? [y/yes]: ").strip().lower()
    except EOFError:
        print("No confirmation received; run cancelled.", flush=True)
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user; run cancelled.", flush=True)
        return False
    if answer not in {"y", "yes"}:
        print("Cancelled by user; run not started.", flush=True)
        return False
    print("", flush=True)
    return True
