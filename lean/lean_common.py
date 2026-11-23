"""Shared helpers for the Lean replay scripts in this directory."""

from pathlib import Path
import subprocess
from time import monotonic

ROOT = Path(__file__).resolve().parent
TOOLCHAIN = (ROOT / "lean-toolchain").read_text(encoding="utf-8").strip()


def run(command, work, env):
    started = monotonic()
    result = subprocess.run(command, cwd=work, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    result.check_returncode()
    print(f"elapsed {monotonic() - started:.3f} seconds")
    return result.stdout
