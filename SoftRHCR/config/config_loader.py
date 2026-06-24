"""Multi-file configuration loader — replaces ``user_config.py``.

Loading pipeline:
  1. ``mode_{kind}.json`` -> algorithm + mode-specific metadata
  2. ``env.json``         -> environment parameters
  3. ``algo.json`` + ``algo_{algorithm}.json`` -> algorithm parameters (deep merge)
  4. CLI overrides        -> routed to env / algo / mode payloads via the routing table
  5. Build ``RunConfig`` and return a metadata dict
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Sequence

from SoftRHCR.config.configBase import (
    EnvConfig,
    RunConfig,
    algo_config_from_dict,
)

ConfigKind = Literal["train", "evaluate", "experiment"]
USERCFG_DIR = Path(__file__).resolve().parent / "usercfg"

_KIND_MAP: dict[str, ConfigKind] = {
    "train": "train", "training": "train",
    "eval": "evaluate", "evaluate": "evaluate", "evaluation": "evaluate",
    "exp": "experiment", "experiment": "experiment", "runtime": "experiment",
}
_ACTIVE_MODE: dict[str, str] = {"train": "train", "evaluate": "eval", "experiment": "runtime"}

# --- CLI argument -> scope routing --------------------------------------------
_ENV_CLI_FIELDS = {
    "num_agvs", "map_size", "fov_size", "kstep_conflict_check",
    "targets_on_shelf", "planner_type",
}
_ALGO_CLI_FIELDS = {
    "lr", "min_lr", "lr_schedule", "horizon_len", "max_train_steps",
    "buffer_size", "batch_size", "save_interval_steps", "gamma",
    "ppo_epoch", "clip_param", "entropy_coef", "value_loss_coef",
    "use_gae", "gae_lambda", "seed", "device",
    "norm_type", "extractor_backbone", "norm_after_concat",
    "profile", "profile_interval_s", "target_update_interval",
    "optimizer", "weight_decay",
}
_MODE_CLI_FIELDS = {
    "run_name", "run_dir", "render", "render_interval_s",
    "model_path", "run_config_path", "eval_episodes", "eval_threads",
    "task_limit", "msgs_mode", "checkpoint_path",
}


# =============================================================================
# JSON loading
# =============================================================================
def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# =============================================================================
# CLI helpers
# =============================================================================
def _coerce_scalar(value: str) -> Any:
    text = str(value).strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _parse_kv_pairs(items: Optional[Iterable[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in items or ():
        raw = str(raw).strip()
        if not raw or "=" not in raw:
            raise ValueError(f"Invalid key=value pair: {raw!r}")
        key, val = raw.split("=", 1)
        result[key.strip()] = _coerce_scalar(val)
    return result


def _bool_action() -> type[argparse.Action]:
    return getattr(argparse, "BooleanOptionalAction", argparse._StoreTrueAction)


def _build_parser(kind: ConfigKind) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config-dir", type=str, default=None,
                        help="config directory path (defaults to usercfg/)")
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--num-agvs", type=int, default=None)
    parser.add_argument("--map-size", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fov-size", type=int, default=None)
    parser.add_argument("--kstep-conflict-check", type=int, default=None)
    parser.add_argument("--targets-on-shelf", action=_bool_action(), default=None)
    parser.add_argument("--planner", type=str, default=None)
    parser.add_argument("--planner-arg", action="append", default=None)
    parser.add_argument("--render", action=_bool_action(), default=None)
    parser.add_argument("--render-interval", type=float, default=None)
    # algo
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--lr-schedule", type=str, default=None)
    parser.add_argument("--horizon-len", type=int, default=None)
    parser.add_argument("--total-train-steps", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-interval-steps", type=int, default=None)
    parser.add_argument("--profile", action=_bool_action(), default=None)
    parser.add_argument("--extractor-backbone", type=str, default=None)
    parser.add_argument("--norm-type", type=str, default=None)
    parser.add_argument("--norm-after-concat", type=str, default=None)
    parser.add_argument("--algo-override", action="append", default=None)
    # mode-specific
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--run-config-path", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-threads", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--msgs-mode", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser


def _parse_cli(kind: ConfigKind, argv: Optional[Sequence[str]] = None
               ) -> tuple[Optional[str], dict[str, Any]]:
    """Return ``(config_dir, overrides)``. Keys in ``overrides`` are normalized."""
    parser = _build_parser(kind)
    ns = parser.parse_args(argv)
    payload: dict[str, Any] = {}
    for key, value in vars(ns).items():
        if value is None:
            continue
        payload[key] = value
    # Normalize key names
    if "max_steps" in payload:
        payload["max_episode_steps"] = payload.pop("max_steps")
    if "render_interval" in payload:
        payload["render_interval_s"] = payload.pop("render_interval")
    if "planner" in payload:
        payload["planner_type"] = payload.pop("planner")
    if "checkpoint" in payload:
        payload["checkpoint_path"] = payload.pop("checkpoint")
    if "total_train_steps" in payload:
        payload["max_train_steps"] = payload.pop("total_train_steps")
    # planner-arg -> planner_overrides
    planner_arg = payload.pop("planner_arg", None)
    if planner_arg:
        payload["_planner_overrides"] = _parse_kv_pairs(planner_arg)
    # algo-override -> _algo_overrides
    algo_override = payload.pop("algo_override", None) or payload.pop("algo_override", None)
    if algo_override:
        payload["_algo_overrides"] = _parse_kv_pairs(algo_override)
    config_dir = payload.pop("config_dir", None)
    return config_dir, payload


# =============================================================================
# Scope routing
# =============================================================================
def _route_overrides(overrides: dict[str, Any],
                     env_payload: dict, algo_payload: dict, mode_payload: dict,
                     kind: ConfigKind) -> None:
    """Inject CLI overrides into the matching payload in place."""
    for key, value in list(overrides.items()):
        if key.startswith("_"):
            continue
        if key in _ENV_CLI_FIELDS:
            # Some env fields are stored per-mode
            if key in ("num_agvs", "map_size", "seed", "max_episode_steps"):
                mode_key = _ACTIVE_MODE[kind]
                env_payload.setdefault(mode_key, {})[key] = value
            else:
                env_payload[key] = value
        elif key in _ALGO_CLI_FIELDS:
            algo_payload[key] = value
        elif key in _MODE_CLI_FIELDS:
            mode_payload[key] = value
        # Unknown keys are silently ignored (or passed explicitly via --algo-override)
    # _planner_overrides
    planner_ovr = overrides.get("_planner_overrides")
    if planner_ovr:
        planner = env_payload.setdefault("planner", {})
        existing = planner.get("overrides", {}) or {}
        existing.update(planner_ovr)
        planner["overrides"] = existing
    # _algo_overrides
    algo_ovr = overrides.get("_algo_overrides")
    if algo_ovr:
        algo_payload.update(algo_ovr)


# =============================================================================
# Run directory
# =============================================================================
def _sanitize(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return text.strip("._-") or "run"


def _auto_run_name(algorithm: str, kind: ConfigKind) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{kind}_{_sanitize(algorithm)}_{ts}"


def _resolve_run_dir(mode_payload: dict, kind: ConfigKind, algorithm: str) -> Path:
    run_dir = str(mode_payload.get("run_dir", "") or "").strip()
    if not run_dir:
        raise ValueError("run_dir must be set")
    run_name = str(mode_payload.get("run_name", "") or "").strip()
    if not run_name or run_name.lower() == "default" or run_name.lower().endswith("_default"):
        run_name = _auto_run_name(algorithm, kind)
        mode_payload["run_name"] = run_name
    root = (Path(run_dir).expanduser() / _sanitize(run_name)).resolve()
    mode_payload["run_dir"] = str(root)
    for sub in ("config", "results", "artifacts"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if kind == "train":
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    return root


# =============================================================================
# Main entry point
# =============================================================================
def load_and_build_config(kind: str, argv: Optional[Sequence[str]] = None
                          ) -> tuple[RunConfig, dict[str, Any]]:
    """Load the multi-file JSON configuration and build a ``RunConfig``.

    Returns ``(run_cfg, meta)``. ``meta`` holds mode-specific metadata such as
    run_name, run_dir, model_path, checkpoint_path and eval_episodes.
    """
    normalized = _KIND_MAP.get(str(kind or "").strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported config kind: {kind!r}")

    # 1. Parse CLI
    config_dir_str, cli_overrides = _parse_cli(normalized, argv)
    config_dir = Path(config_dir_str) if config_dir_str else USERCFG_DIR

    # 2. Load JSON files
    mode_payload = _load_json(config_dir / f"mode_{normalized}.json")
    env_payload = _load_json(config_dir / "env.json")
    algo_base = _load_json(config_dir / "algo.json")

    # Resolve the algorithm (CLI > mode JSON > default)
    algorithm = str(cli_overrides.get("algorithm",
                    mode_payload.get("algorithm", "mappo"))).strip().lower()
    algo_specific = _load_json(config_dir / f"algo_{algorithm}.json")
    algo_payload = _deep_merge(algo_base, algo_specific)

    # 3. Apply CLI overrides
    _route_overrides(cli_overrides, env_payload, algo_payload, mode_payload, normalized)

    # 4. Handle rendering
    active_mode = _ACTIVE_MODE[normalized]
    if mode_payload.get("render", False):
        env_mode = env_payload.setdefault(active_mode, {})
        env_mode["render_mode"] = "human"
        if "render_interval_s" in mode_payload:
            env_mode["render_interval_s"] = float(mode_payload["render_interval_s"])

    # 5. Build EnvConfig
    env_cfg = EnvConfig.from_dict(env_payload)

    # 6. Build algo config
    algo_cfg = algo_config_from_dict(algorithm, algo_payload)

    # 7. Resolve the output directory
    root_dir = _resolve_run_dir(mode_payload, normalized, algorithm)
    if normalized == "train" and hasattr(algo_cfg, "model_dir"):
        algo_cfg.model_dir = str(root_dir / "checkpoints")

    # 8. Assemble RunConfig
    run_cfg = RunConfig(algorithm=algorithm, env=env_cfg, algo=algo_cfg, observation={})

    # 9. Build the meta dict.
    # eval_episodes is read from the active mode's env sub-object.
    env_mode_dict = env_payload.get(active_mode, {})
    if "eval_episodes" in env_mode_dict:
        env_cfg.eval_episodes = max(1, int(env_mode_dict["eval_episodes"]))
    meta: dict[str, Any] = {
        "run_name": mode_payload.get("run_name", ""),
        "run_dir": str(root_dir),
        "device": str(algo_payload.get("device", "auto")),
        "render_interval_s": float(mode_payload.get("render_interval_s", 0.0)),
        "eval_episodes": int(env_cfg.eval_episodes),
    }
    # Mode-specific fields
    for key in ("model_path", "run_config_path", "checkpoint_path",
                "eval_threads", "task_limit", "msgs_mode",
                "start_episode", "total_steps"):
        if key in mode_payload:
            meta[key] = mode_payload[key]

    return run_cfg, meta
