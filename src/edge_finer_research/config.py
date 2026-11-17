from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file without introducing a YAML runtime dependency."""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(value: str, *, base_dir: str | Path) -> Path:
    """Resolve a configured path relative to the config location."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (Path(base_dir) / candidate).resolve()

