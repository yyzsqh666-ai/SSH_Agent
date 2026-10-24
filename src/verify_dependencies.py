"""Check isolated Python, numerical libraries, headless plotting and pytest."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import site
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON output file; existing files are preserved.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file name.")
    root = Path(__file__).resolve().parents[1]
    os.environ["MPLCONFIGDIR"] = str(root / ".matplotlib")
    os.environ["MPLBACKEND"] = "Agg"

    import numpy as np
    import scipy.linalg
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import pytest

    expected_prefix = (root / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected_prefix or sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this check using the project's .venv Python.")
    if site.ENABLE_USER_SITE is not False:
        raise RuntimeError("User site-packages must be disabled in the project environment.")
    for module in (np, scipy.linalg, matplotlib, pytest):
        if not Path(module.__file__).resolve().is_relative_to(expected_prefix):
            raise RuntimeError(f"Dependency outside project environment: {module.__file__}")

    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    np.testing.assert_allclose(np.linalg.eigvalsh(matrix), [-1.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(scipy.linalg.eigvalsh(matrix), [-1.0, 3.0], atol=1e-12)
    fig = Figure(figsize=(2, 2))
    canvas = FigureCanvasAgg(fig)
    fig.subplots().plot([0, 1], [0, 1])
    canvas.draw()
    if canvas.buffer_rgba().nbytes == 0:
        raise RuntimeError("Headless plotting produced an empty canvas.")
    command = [sys.executable, "-B", "-m", "pytest", "--version"]
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if process.returncode:
        raise RuntimeError(f"pytest startup failed: {process.stdout}\n{process.stderr}")
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-B", str(Path(__file__).resolve()),
                    "--output", str(args.output)],
        "python_version": sys.version,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "matplotlib", "pytest")},
        "checks": {
            "project_environment_isolation": "passed",
            "all_four_imports": "passed",
            "numpy_known_matrix_eigenvalues": "passed",
            "scipy_known_matrix_eigenvalues": "passed",
            "matplotlib_agg_render": "passed",
            "pytest_startup": "passed",
        },
        "numerical_check": {"matrix": matrix.tolist(), "expected_eigenvalues": [-1, 3],
                            "absolute_tolerance": 1e-12},
        "pytest_command": command,
        "pytest_stdout": process.stdout,
        "pytest_stderr": process.stderr,
        "physics_tests_run": False,
        "note": "Environment verification only; no SSH calculation or literature search.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
