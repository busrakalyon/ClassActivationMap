from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str], root: Path, *, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast-forward safely, preserving ignored results")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--then", choices=("none", "tests", "smoke"), default="none")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tracked_changes = run(
        ["git", "status", "--porcelain", "--untracked-files=no"], root, capture=True
    )
    if tracked_changes:
        raise RuntimeError(
            "Tracked local changes exist; refusing to pull so they are not overwritten:\n"
            + tracked_changes
        )
    run(["git", "fetch", args.remote, args.branch], root)
    run(["git", "merge", "--ff-only", f"{args.remote}/{args.branch}"], root)
    if args.then == "tests":
        run(["uv", "run", "python", "-m", "unittest", "discover", "-s", "tests", "-v"], root)
    elif args.then == "smoke":
        run(
            [
                "uv",
                "run",
                "python",
                "scripts/smoke_test.py",
                "--device",
                args.device,
            ],
            root,
        )


if __name__ == "__main__":
    main()
