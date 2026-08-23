from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humanitarian_forecast.core.paths import PATHS


INFO_FILENAME = "info.blt"
INFO_FORMAT = "humanitarian-forecast-model-info/v1"


@dataclass
class ModelInfo:
    subsystem: str
    version: str
    status: str
    description: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    format: str = INFO_FORMAT


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_info(model_dir: Path, info: ModelInfo | dict[str, Any]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(info) if isinstance(info, ModelInfo) else dict(info)
    payload.setdefault("format", INFO_FORMAT)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload["artifacts"] = sorted(
        p.name for p in model_dir.iterdir() if p.is_file() and p.name != INFO_FILENAME
    )
    path = model_dir / INFO_FILENAME
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_info(model_dir: Path) -> dict[str, Any]:
    path = model_dir / INFO_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def discover_models(root: Path | None = None) -> list[tuple[Path, dict[str, Any]]]:
    root = root or PATHS.models
    found: list[tuple[Path, dict[str, Any]]] = []
    if not root.exists():
        return found
    for path in sorted(root.rglob(INFO_FILENAME)):
        try:
            found.append((path.parent, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return found


def metric_payload(model_dir: Path) -> dict[str, Any]:
    """Collect existing JSON metrics without inventing a score for old artifacts."""
    result: dict[str, Any] = {}
    for path in sorted(model_dir.glob("*.json")):
        if "metric" not in path.name and "calibration" not in path.name:
            continue
        try:
            result[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return result
