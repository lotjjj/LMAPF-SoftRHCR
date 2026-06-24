from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from typing import Any, Optional, Type

from SoftRHCR.config.configBase import (
    CommonConfig,
    FollowPlannerConfig,
    GateBlendConfig,
    GateBlendMAPPOConfig,
    IPPOConfig,
    MAPPOConfig,
    SoftRHCRConfig,
    SoftRHCRMAPPOConfig,
)


@dataclass(frozen=True)
class ExtractorRegistryEntry:
    backbone: str
    spatial: str
    description: str = ""


@dataclass(frozen=True)
class AlgorithmRegistryEntry:
    name: str
    config_cls: Type[CommonConfig]
    agent_path: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    select_action_with_info: bool = False
    agent_class_names: tuple[str, ...] = ()


def _load_symbol(path: str) -> Any:
    module_name, attr_name = str(path).split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _load_network_backbones() -> tuple[str, ...]:
    try:
        from SoftRHCR.modules.network import available_backbones

        return tuple(str(x) for x in available_backbones())
    except Exception:
        return ()


def _parse_backbone(backbone: str) -> str:
    text = str(backbone or "").strip().lower()
    return text or "default"


def _extractor_description(spatial: str) -> str:
    return spatial


def extractor_registry_entries() -> tuple[ExtractorRegistryEntry, ...]:
    entries: list[ExtractorRegistryEntry] = []
    for item in _load_network_backbones():
        spatial = _parse_backbone(item)
        entries.append(
            ExtractorRegistryEntry(
                backbone=str(item),
                spatial=str(spatial),
                description=_extractor_description(str(spatial)),
            )
        )
    return tuple(entries)


def extractor_options() -> tuple[str, ...]:
    return tuple(entry.backbone for entry in extractor_registry_entries())


def normalize_extractor_token(backbone: Any) -> Optional[str]:
    text = str(backbone or "").strip().lower()
    if text == "":
        return None
    return text


def extractor_registry_snapshot() -> dict[str, Any]:
    extractors: list[dict[str, Any]] = []
    for entry in extractor_registry_entries():
        extractors.append(
            {
                "backbone": entry.backbone,
                "spatial": entry.spatial,
                "description": entry.description,
            }
        )
    return {"extractors": extractors}


def extractor_registry_signature() -> str:
    snapshot = extractor_registry_snapshot()
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)


def _algorithm_registry_entries() -> tuple[AlgorithmRegistryEntry, ...]:
    return (
        AlgorithmRegistryEntry(
            name="gateblend_mappo",
            config_cls=GateBlendMAPPOConfig,
            agent_path="SoftRHCR.algorithms.GateBlend.gate_blend_mappo:GateBlendMAPPOAgent",
            description="GateBlend with MAPPO critic",
            agent_class_names=("GateBlendMAPPOAgent",),
        ),
        AlgorithmRegistryEntry(
            name="gateblend",
            config_cls=GateBlendConfig,
            agent_path="SoftRHCR.algorithms.GateBlend.gate_blend:GateBlendAgent",
            description="GateBlend with IPPO actor",
            agent_class_names=("GateBlendAgent",),
        ),
        AlgorithmRegistryEntry(
            name="soft_rhcr_mappo",
            config_cls=SoftRHCRMAPPOConfig,
            agent_path="SoftRHCR.algorithms.SoftRHCR.soft_rhcr_mappo:SoftRHCRMAPPOAgent",
            description="SoftRHCR with MAPPO critic",
            select_action_with_info=True,
            agent_class_names=("SoftRHCRMAPPOAgent",),
        ),
        AlgorithmRegistryEntry(
            name="soft_rhcr",
            config_cls=SoftRHCRConfig,
            agent_path="SoftRHCR.algorithms.SoftRHCR.soft_rhcr:SoftRHCRAgent",
            description="SoftRHCR with IPPO actor",
            select_action_with_info=True,
            agent_class_names=("SoftRHCRAgent",),
        ),

        AlgorithmRegistryEntry(
            name="mappo",
            config_cls=MAPPOConfig,
            agent_path="SoftRHCR.algorithms.MAPPO.mappo:MAPPOAgent",
            description="MAPPO",
            agent_class_names=("MAPPOAgent",),
        ),
        AlgorithmRegistryEntry(
            name="ippo",
            config_cls=IPPOConfig,
            agent_path="SoftRHCR.algorithms.IPPO.ippo:IPPOAgent",
            description="IPPO",
            agent_class_names=("IPPOAgent",),
        ),
        AlgorithmRegistryEntry(
            name="follow_planner",
            config_cls=FollowPlannerConfig,
            agent_path="SoftRHCR.algorithms.FollowPlanner.follow_planner:FollowPlannerAgent",
            description="Follow planner only",
            agent_class_names=("FollowPlannerAgent",),
        ),
    )


def algorithm_registry_entries() -> tuple[AlgorithmRegistryEntry, ...]:
    return _algorithm_registry_entries()


def normalize_algorithm_token(algorithm: Any) -> Optional[str]:
    text = str(algorithm or "").strip().lower()
    if text == "":
        return None
    for entry in algorithm_registry_entries():
        if text == entry.name or text in entry.aliases:
            return entry.name
    return text


def algorithm_options(*, include_aliases: bool = False) -> tuple[str, ...]:
    options: list[str] = []
    for entry in algorithm_registry_entries():
        options.append(entry.name)
        if include_aliases:
            options.extend(entry.aliases)
    return tuple(options)


def get_algorithm_entry(algorithm: Any) -> Optional[AlgorithmRegistryEntry]:
    normalized = normalize_algorithm_token(algorithm)
    if normalized is None:
        return None
    for entry in algorithm_registry_entries():
        if entry.name == normalized:
            return entry
    return None


def get_algorithm_entry_from_config(config: Any) -> Optional[AlgorithmRegistryEntry]:
    for entry in algorithm_registry_entries():
        if isinstance(config, entry.config_cls):
            return entry
    return None


def get_algorithm_entry_from_agent(agent: Any) -> Optional[AlgorithmRegistryEntry]:
    agent_name = type(agent).__name__
    for entry in algorithm_registry_entries():
        if agent_name in entry.agent_class_names:
            return entry
    return None


def algorithm_name_from_agent(agent: Any) -> str:
    entry = get_algorithm_entry_from_agent(agent)
    if entry is not None:
        return entry.name
    return str(type(agent).__name__).lower()


def agent_select_action_requires_info(agent: Any) -> bool:
    entry = get_algorithm_entry_from_agent(agent)
    return bool(entry.select_action_with_info) if entry is not None else False


def _prepare_legacy_algorithm_config(algo: str, data: Optional[dict[str, Any]]) -> dict[str, Any]:
    cfg_data = dict(data or {})
    if "reward_mode" not in cfg_data:
        # Apply legacy conversion only when the legacy field intrinsic_reward_enabled is present
        legacy_intrinsic_enabled = cfg_data.pop("intrinsic_reward_enabled", None)
        legacy_intrinsic_mode = str(cfg_data.pop("intrinsic_reward_mode", "aggressive") or "aggressive").strip().lower()
        if legacy_intrinsic_enabled is not None:
            # Set reward_mode only when the legacy field exists
            intrinsic_enabled = bool(legacy_intrinsic_enabled)
            cfg_data["reward_mode"] = legacy_intrinsic_mode if intrinsic_enabled else "legacy"
        # else: leave reward_mode unset and let the dataclass fall back to its own default
    else:
        cfg_data["reward_mode"] = str(cfg_data.get("reward_mode", "legacy") or "legacy").strip().lower()
    if algo in ("soft_rhcr", "soft_rhcr_mappo"):
        legacy_steps = cfg_data.pop("force_rl_prob_decay_steps", None)
        if "decay_steps_ratio" not in cfg_data and legacy_steps is not None:
            try:
                max_train_steps = int(cfg_data.get("max_train_steps", CommonConfig.max_train_steps))
            except Exception:
                max_train_steps = int(CommonConfig.max_train_steps)
            if max_train_steps > 0:
                cfg_data["decay_steps_ratio"] = float(legacy_steps) / float(max_train_steps)
            else:
                cfg_data["decay_steps_ratio"] = 0.0
    if algo in ("soft_rhcr", "soft_rhcr_mappo"):
        if "planner_aux_loss" not in cfg_data:
            has_legacy_kl = ("kl_coef" in cfg_data) or ("kl_coef_end" in cfg_data)
            has_consistency = ("fp_consistency_coef" in cfg_data) or ("fp_consistency_coef_end" in cfg_data)
            if has_legacy_kl and not has_consistency:
                cfg_data["planner_aux_loss"] = "kl"
    return cfg_data


def _filter_dataclass_kwargs(cls: Type[Any], data: Optional[dict[str, Any]]) -> dict[str, Any]:
    raw = dict(data or {})
    fields = getattr(cls, "__dataclass_fields__", {})
    return {k: v for k, v in raw.items() if k in fields}


def algo_config_from_dict(algorithm: Any, data: Optional[dict[str, Any]]) -> CommonConfig:
    normalized = normalize_algorithm_token(algorithm) or "mappo"
    entry = get_algorithm_entry(normalized)
    cfg_data = _prepare_legacy_algorithm_config(normalized, data)
    cls = entry.config_cls if entry is not None else CommonConfig
    return cls(**_filter_dataclass_kwargs(cls, cfg_data))


def create_agent(algo_cfg: Any, obs_info: dict[str, Any], action_dim: int, n_agents: int) -> Any:
    entry = get_algorithm_entry_from_config(algo_cfg)
    if entry is None:
        raise ValueError(f"Unsupported algorithm config type: {type(algo_cfg)}")
    agent_cls = _load_symbol(entry.agent_path)
    return agent_cls(algo_cfg, obs_info, action_dim, n_agents=n_agents)


def algorithm_registry_snapshot() -> dict[str, Any]:
    algorithms: list[dict[str, Any]] = []
    for entry in algorithm_registry_entries():
        algorithms.append(
            {
                "name": entry.name,
                "config_cls": entry.config_cls.__name__,
                "agent_path": entry.agent_path,
                "aliases": list(entry.aliases),
                "description": entry.description,
                "select_action_with_info": bool(entry.select_action_with_info),
                "agent_class_names": list(entry.agent_class_names),
            }
        )
    return {"algorithms": algorithms}


def algorithm_registry_signature() -> str:
    snapshot = algorithm_registry_snapshot()
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
