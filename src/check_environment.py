"""Inventory the active Python and installed dependencies; no SSH calculations."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def collect_environment(gpd_python):
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "pytest"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = {
            "version": version,
            "module_found": importlib.util.find_spec(name) is not None,
        }
    gpd = {"python": str(gpd_python), "available": False}
    if gpd_python.is_file():
        command = [str(gpd_python), "-B", "-m", "gpd.runtime_cli",
                   "--runtime", "codex", "--config-dir",
                   str(Path.home() / ".codex"), "--install-scope", "global",
                   "--version"]
        gpd["command"] = command
        try:
            process = subprocess.run(command, capture_output=True, text=True,
                                     timeout=30, check=False)
            gpd.update(exit_code=process.returncode, stdout=process.stdout,
                       stderr=process.stderr, available=process.returncode == 0)
        except (OSError, subprocess.TimeoutExpired) as error:
            gpd["error"] = str(error)
    else:
        gpd["error"] = "GPD Python executable not found at the supplied path."
    missing = [name for name, info in packages.items()
               if info["version"] is None or not info["module_found"]]
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "command_paths": {name: shutil.which(name)
                          for name in ("python", "python3", "py", "git", "gpd")},
        "packages": packages,
        "missing_dependencies": missing,
        "dependency_readiness": "not_ready" if missing else "discovered_unverified",
        "check_limit": "Inventory only; module presence does not test imports or calculations.",
        "gpd": gpd,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpd-python", type=Path,
                        default=Path.home() / ".gpd" / "venv" / "Scripts" / "python.exe")
    parser.add_argument("--output", type=Path,
                        help="Save JSON to a new file; never overwrite an existing file.")
    args = parser.parse_args()
    report = json.dumps(collect_environment(args.gpd_python), ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
