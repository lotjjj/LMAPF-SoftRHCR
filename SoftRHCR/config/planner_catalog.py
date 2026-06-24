from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Optional


@dataclass(frozen=True)
class PlannerFieldSpec:
    key: str
    label: str
    default: Any
    options: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class PlannerCatalogEntry:
    planner_type: str
    fields: tuple[PlannerFieldSpec, ...]


def _load_simulator_registry() -> dict[str, Any]:
    try:
        from LMAPFEnv.configBase import PLANNER_REGISTRY
    except Exception:
        return {}
    return dict(PLANNER_REGISTRY or {})


def planner_catalog_entries() -> tuple[PlannerCatalogEntry, ...]:
    entries: list[PlannerCatalogEntry] = []
    for planner_type, spec in _load_simulator_registry().items():
        fields = tuple(
            PlannerFieldSpec(
                key=str(param.key),
                label=str(param.label),
                default=param.default,
                options=tuple(str(option) for option in getattr(param, "options", ()) or ()),
                description=str(getattr(param, "description", "") or ""),
            )
            for param in getattr(spec, "params", ()) or ()
        )
        entries.append(PlannerCatalogEntry(planner_type=str(planner_type), fields=fields))
    return tuple(entries)


def planner_options(*, include_none: bool = False) -> tuple[str, ...]:
    options = [entry.planner_type for entry in planner_catalog_entries()]
    if include_none:
        return ("none", *options)
    return tuple(options)


def get_planner_entry(planner_type: Any) -> Optional[PlannerCatalogEntry]:
    normalized = normalize_planner_type_token(planner_type)
    if normalized is None:
        return None
    for entry in planner_catalog_entries():
        if entry.planner_type == normalized:
            return entry
    return None


def planner_field_specs(planner_type: Any) -> tuple[PlannerFieldSpec, ...]:
    entry = get_planner_entry(planner_type)
    return () if entry is None else entry.fields


def default_planner_overrides(planner_type: Any) -> dict[str, Any]:
    return {field.key: field.default for field in planner_field_specs(planner_type)}


def normalize_planner_type_token(planner_type: Any) -> Optional[str]:
    text = str(planner_type or "").strip()
    if text == "" or text.lower() == "none":
        return None
    return text


def planner_registry_snapshot() -> dict[str, Any]:
    planners: list[dict[str, Any]] = []
    for entry in planner_catalog_entries():
        planners.append(
            {
                "planner_type": entry.planner_type,
                "fields": [
                    {
                        "key": field.key,
                        "label": field.label,
                        "default": field.default,
                        "options": list(field.options),
                        "description": field.description,
                    }
                    for field in entry.fields
                ],
            }
        )
    return {"planners": planners}


def planner_registry_signature() -> str:
    snapshot = planner_registry_snapshot()
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
