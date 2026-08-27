import copy
import json
from pathlib import Path
from typing import Any

EARLYSELECTKV_METHOD_ALIASES = {
    "lookaheadkv": "earlyselectkv",
    "lookaheadkv_in": "earlyselectkv_in",
    "lookahead_in_mid": "earlyselectkv_in_mid",
    "lookahead_in_mid_local": "earlyselectkv_in_mid_local",
    "lookaheadkv_topk": "earlyselectkv_topk",
    "lookaheadkv_local": "earlyselectkv_local",
    "lookaheadkv_hsa_local": "earlyselectkv_hsa_local",
    "lookahead_quest": "earlyselect_quest",
    "lookahead_loki": "earlyselect_loki",
    "lookaheadkv_mid_svd_r64": "earlyselectkv_mid_svd_r64",
    "lookaheadkv_mid_svd_r128": "earlyselectkv_mid_svd_r128",
    "lookaheadkv_mid_svd_r256": "earlyselectkv_mid_svd_r256",
    "lookaheadkv_mid_svd_r512": "earlyselectkv_mid_svd_r512",
}


def canonical_method_key(method: str) -> str:
    """Return the current config key for a possibly legacy method name."""

    method = str(method)
    normalized = method.lower()
    if normalized in EARLYSELECTKV_METHOD_ALIASES:
        return EARLYSELECTKV_METHOD_ALIASES[normalized]
    if normalized.startswith("earlyselect"):
        return normalized
    return method


def method_config_keys(method: str) -> tuple[str, ...]:
    """Return method config keys to keep/read, with the caller's spelling first."""

    method = str(method)
    canonical = canonical_method_key(method)
    keys = [method]
    if canonical not in keys:
        keys.append(canonical)
    keys.extend(
        legacy
        for legacy, current in EARLYSELECTKV_METHOD_ALIASES.items()
        if current == canonical and legacy not in keys
    )
    return tuple(keys)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_json_with_extends(path: str | Path) -> dict[str, Any]:
    """Load a JSON config, resolving optional relative `extends` entries."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    extends = config.get("extends", [])
    if isinstance(extends, str):
        extends = [extends]

    merged: dict[str, Any] = {}
    for base_path in extends:
        resolved_base = (config_path.parent / base_path).resolve()
        merged = _deep_merge(merged, load_json_with_extends(resolved_base))

    return _deep_merge(merged, config)
