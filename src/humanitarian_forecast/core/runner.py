from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

from humanitarian_forecast.core.paths import PATHS


def module_command(module: str, args: Iterable[str] = ()) -> list[str]:
    return [sys.executable, "-m", module, *[str(arg) for arg in args]]


def display_command(command: Iterable[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in command)


def run_module(
    module: str,
    args: Iterable[str] = (),
    *,
    dry_run: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    command = module_command(module, args)
    print(f"$ {display_command(command)}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PATHS.src) + (os.pathsep + existing if existing else "")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    subprocess.run(command, cwd=PATHS.root, env=env, check=True)


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
