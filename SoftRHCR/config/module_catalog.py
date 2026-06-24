from __future__ import annotations

import json
from typing import Any, Optional

from SoftRHCR.config.registry import (
    ExtractorRegistryEntry as ModuleCatalogEntry,
    extractor_options,
    extractor_registry_entries,
    extractor_registry_snapshot,
    normalize_extractor_token,
)


def module_catalog_entries() -> tuple[ModuleCatalogEntry, ...]:
    return tuple(extractor_registry_entries())


def module_options() -> tuple[str, ...]:
    return tuple(extractor_options())


def normalize_module_token(backbone: Any) -> Optional[str]:
    return normalize_extractor_token(backbone)


def module_registry_snapshot() -> dict[str, Any]:
    snapshot = extractor_registry_snapshot()
    return {"modules": list(snapshot.get("extractors", []))}


def module_registry_signature() -> str:
    snapshot = module_registry_snapshot()
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
