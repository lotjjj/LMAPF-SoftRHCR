from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from SoftRHCR.config.configBase import (
    EnvConfig,
    RunConfig,
    SoftRHCRConfig,
    SoftRHCRMAPPOConfig,
    algo_config_from_dict,
)
from SoftRHCR.config.registry import algorithm_registry_entries, create_agent as registry_create_agent, normalize_algorithm_token


@dataclass(frozen=True)
class ModelLoadSpec:
    model_path: Path
    run_config_path: Optional[Path]
    algorithm: str
    algo_cfg_payload: dict[str, Any]
    checkpoint_meta: dict[str, Any]
    source: str
    msgs_mode_hint: Optional[str] = None
    msgs_mode_explicit: bool = False


def safe_read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_algorithm_from_filename(model_path: Path) -> str:
    stem = str(model_path.stem or "").strip().lower()
    compact_stem = re.sub(r"[^a-z0-9]+", "", stem)
    candidates: list[tuple[int, str, str]] = []
    for entry in algorithm_registry_entries():
        names = [entry.name, *entry.aliases, *entry.agent_class_names]
        seen: set[str] = set()
        for raw_name in names:
            normalized = str(raw_name or "").strip().lower()
            if normalized == "" or normalized in seen:
                continue
            seen.add(normalized)
            compact_name = re.sub(r"[^a-z0-9]+", "", normalized)
            if compact_name == "":
                continue
            candidates.append((len(compact_name), compact_name, entry.name))
    candidates.sort(reverse=True)
    for _, compact_name, algorithm_name in candidates:
        if compact_name in compact_stem:
            return algorithm_name
    return "unknown"


def _normalize_algorithm_or_none(value: Any) -> Optional[str]:
    normalized = normalize_algorithm_token(value)
    if normalized is None:
        return None
    normalized = str(normalized).strip().lower()
    if normalized in ("", "unknown"):
        return None
    return normalized


def _algorithm_from_agent_class(agent_class: Any) -> Optional[str]:
    agent_name = str(agent_class or "").strip()
    if agent_name == "":
        return None
    for entry in algorithm_registry_entries():
        if agent_name in entry.agent_class_names:
            return entry.name
    return None


def _normalize_msgs_mode_or_none(value: Any) -> Optional[str]:
    mode = str(value or "").strip().lower()
    if mode in ("single", "dual"):
        return mode
    return None


def _extract_msgs_mode_hint(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    direct_mode = _normalize_msgs_mode_or_none(payload.get("soft_rhcr_msgs_mode"))
    if direct_mode is not None:
        return direct_mode
    direct_mode = _normalize_msgs_mode_or_none(payload.get("msgs_mode"))
    if direct_mode is not None:
        return direct_mode
    for nested_key in ("algo", "algo_cfg", "observation"):
        nested = payload.get(nested_key)
        nested_mode = _extract_msgs_mode_hint(nested)
        if nested_mode is not None:
            return nested_mode
    return None


def _read_checkpoint_meta(model_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(str(model_path), map_location="cpu")
    except Exception:
        return {}
    if not isinstance(checkpoint, dict):
        return {}
    meta = checkpoint.get("meta", {})
    return dict(meta) if isinstance(meta, dict) else {}


def resolve_model_load_spec(model_path: str | Path, run_config_path: Optional[str | Path] = None) -> ModelLoadSpec:
    model_path = Path(model_path)
    run_config_file = Path(run_config_path) if run_config_path else None
    checkpoint_meta = _read_checkpoint_meta(model_path)

    meta_run_cfg = checkpoint_meta.get("run_config")
    if not isinstance(meta_run_cfg, dict):
        meta_run_cfg = None
    file_run_cfg = safe_read_json(run_config_file) if run_config_file is not None and run_config_file.exists() else None

    algorithm: Optional[str] = _normalize_algorithm_or_none(checkpoint_meta.get("algorithm"))
    source = "checkpoint"
    if algorithm is None and isinstance(meta_run_cfg, dict):
        algorithm = _normalize_algorithm_or_none(meta_run_cfg.get("algorithm"))
    if algorithm is None:
        algorithm = _algorithm_from_agent_class(checkpoint_meta.get("agent_class"))
    if algorithm is None and isinstance(file_run_cfg, dict):
        try:
            algorithm = RunConfig.from_dict(file_run_cfg).algorithm
            source = "run_config"
        except Exception:
            algorithm = None
    if algorithm is None:
        algorithm = _normalize_algorithm_or_none(_infer_algorithm_from_filename(model_path)) or "unknown"
        source = "filename"

    algo_cfg_payload = checkpoint_meta.get("algo_cfg")
    if not isinstance(algo_cfg_payload, dict) and isinstance(meta_run_cfg, dict):
        algo_cfg_payload = meta_run_cfg.get("algo", meta_run_cfg.get("algo_cfg", {}))
    if not isinstance(algo_cfg_payload, dict) and isinstance(file_run_cfg, dict):
        algo_cfg_payload = file_run_cfg.get("algo", file_run_cfg.get("algo_cfg", {}))
    if not isinstance(algo_cfg_payload, dict):
        algo_cfg_payload = {}

    msgs_mode_hint = _normalize_msgs_mode_or_none(algo_cfg_payload.get("soft_rhcr_msgs_mode"))
    msgs_mode_explicit = msgs_mode_hint is not None
    if not msgs_mode_explicit and isinstance(meta_run_cfg, dict):
        msgs_mode_hint = _extract_msgs_mode_hint(meta_run_cfg)
        msgs_mode_explicit = msgs_mode_hint is not None
    if not msgs_mode_explicit and isinstance(file_run_cfg, dict):
        msgs_mode_hint = _extract_msgs_mode_hint(file_run_cfg)
        msgs_mode_explicit = msgs_mode_hint is not None

    return ModelLoadSpec(
        model_path=model_path,
        run_config_path=run_config_file,
        algorithm=str(algorithm or "unknown"),
        algo_cfg_payload=dict(algo_cfg_payload),
        checkpoint_meta=checkpoint_meta,
        source=source,
        msgs_mode_hint=msgs_mode_hint,
        msgs_mode_explicit=msgs_mode_explicit,
    )


def build_algo_config_from_model_spec(
    spec: ModelLoadSpec,
    device: str = "auto",
    env_cfg: Optional[EnvConfig] = None,
    override_msgs_mode: Optional[str] = None,
):
    algo_cfg = algo_config_from_dict(spec.algorithm, spec.algo_cfg_payload)
    algo_cfg.device = str(device or "auto").strip() or "auto"
    if (
        override_msgs_mode in ("single", "dual")
        and isinstance(algo_cfg, (SoftRHCRConfig, SoftRHCRMAPPOConfig))
    ):
        setattr(algo_cfg, "soft_rhcr_msgs_mode", str(override_msgs_mode))
    try:
        if isinstance(algo_cfg, (SoftRHCRConfig, SoftRHCRMAPPOConfig)) and env_cfg is not None:
            from SoftRHCR.algorithms.SoftRHCR.rhcr_utils import apply_soft_rhcr_defaults

            apply_soft_rhcr_defaults(env_cfg, algo_cfg)
    except Exception:
        pass
    return algo_cfg


def create_agent_from_model_spec(
    spec: ModelLoadSpec,
    obs_info: dict[str, Any],
    action_dim: int,
    n_agents: int,
    device: str = "auto",
    env_cfg: Optional[EnvConfig] = None,
    override_msgs_mode: Optional[str] = None,
):
    algo_cfg = build_algo_config_from_model_spec(
        spec,
        device=device,
        env_cfg=env_cfg,
        override_msgs_mode=override_msgs_mode,
    )
    return registry_create_agent(algo_cfg, obs_info, action_dim, n_agents)


def load_agent_state(agent: Any, model_path: str | Path, evaluation: bool = False) -> None:
    if evaluation:
        setattr(agent, "_eval_skip_value", True)
        try:
            agent.load_model(str(model_path), load_critic=False, load_optimizer=False)
        except TypeError:
            agent.load_model(str(model_path))
    else:
        agent.load_model(str(model_path))
    if evaluation:
        for attr in ("force_rl_prob_start", "force_rl_prob_end", "last_force_rl_prob"):
            if hasattr(agent, attr):
                try:
                    setattr(agent, attr, 0.0)
                except Exception:
                    pass
        if hasattr(agent, "set_training_mode"):
            try:
                agent.set_training_mode(False)
            except Exception:
                pass


def load_training_checkpoint(
    agent: Any,
    model_path: str | Path,
    run_config_path: Optional[str | Path] = None,
    expected_algorithm: Optional[str] = None,
) -> ModelLoadSpec:
    spec = resolve_model_load_spec(model_path, run_config_path)
    normalized_expected = _normalize_algorithm_or_none(expected_algorithm)
    normalized_actual = _normalize_algorithm_or_none(spec.algorithm)
    if (
        normalized_expected is not None
        and normalized_actual is not None
        and normalized_expected != normalized_actual
    ):
        raise ValueError(
            f"Algorithm mismatch: checkpoint algorithm={normalized_actual} "
            f"does not match the current training algorithm={normalized_expected}"
        )
    load_agent_state(agent, spec.model_path, evaluation=False)
    return spec


def _soft_rhcr_mode_candidates(spec: ModelLoadSpec) -> list[Optional[str]]:
    if spec.algorithm not in ("soft_rhcr", "soft_rhcr_mappo"):
        return [None]
    explicit_mode = _normalize_msgs_mode_or_none(spec.msgs_mode_hint)
    if spec.msgs_mode_explicit and explicit_mode is not None:
        return [explicit_mode]
    return ["single", "dual"]


def create_and_load_agent_from_model_spec(
    spec: ModelLoadSpec,
    obs_info: dict[str, Any],
    action_dim: int,
    n_agents: int,
    device: str = "auto",
    env_cfg: Optional[EnvConfig] = None,
    evaluation: bool = False,
):
    errors: list[tuple[Optional[str], Exception]] = []
    for candidate_mode in _soft_rhcr_mode_candidates(spec):
        try:
            agent = create_agent_from_model_spec(
                spec,
                obs_info,
                action_dim,
                n_agents,
                device=device,
                env_cfg=env_cfg,
                override_msgs_mode=candidate_mode,
            )
            load_agent_state(agent, spec.model_path, evaluation=evaluation)
            if candidate_mode in ("single", "dual"):
                try:
                    setattr(agent, "_loaded_soft_rhcr_msgs_mode", candidate_mode)
                except Exception:
                    pass
            return agent
        except Exception as exc:
            errors.append((candidate_mode, exc))
    if len(errors) == 1:
        raise errors[0][1]
    attempted = ", ".join(str(mode or "default") for mode, _ in errors)
    detail = " | ".join(f"{mode or 'default'}: {exc}" for mode, exc in errors)
    raise RuntimeError(f"Model loading failed; tried {attempted} in order. Details: {detail}")
