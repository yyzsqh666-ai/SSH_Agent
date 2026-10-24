"""Standard 1D SSH bands: independent Hermitian eigensolve and analytic check."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


CASES = ((1.5, 0.5), (0.5, 1.5), (1.0, 1.0))
ATOL = 1e-12
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _validate_hoppings(t1, t2):
    if not all(np.isfinite(t) and t > 0 for t in (t1, t2)):
        raise ValueError("t1 and t2 must be finite positive real numbers.")


def k_grid(num_k=1001):
    """Include both zone edges so the positive-hopping gap minimum is sampled."""
    if not isinstance(num_k, (int, np.integer)) or num_k < 3:
        raise ValueError("num_k must be an integer >= 3.")
    return np.linspace(-np.pi, np.pi, num_k, endpoint=True)


def bloch_hamiltonian(k, t1, t2):
    """Return (..., 2, 2) matrices in the (A, B) basis, with a=1."""
    _validate_hoppings(t1, t2)
    k = np.asarray(k, dtype=float)
    if not np.all(np.isfinite(k)):
        raise ValueError("k must be finite.")
    hopping = t1 + t2 * np.exp(-1j * k)
    hamiltonian = np.zeros(k.shape + (2, 2), dtype=complex)
    hamiltonian[..., 0, 1] = hopping
    hamiltonian[..., 1, 0] = hopping.conjugate()
    return hamiltonian


def numerical_bands(k, t1, t2):
    """Solve H with numpy.linalg.eigh; do not use the analytic band formula."""
    eigenvalues, _eigenvectors = np.linalg.eigh(bloch_hamiltonian(k, t1, t2))
    return eigenvalues  # Ascending order: lower band, upper band.


def analytic_bands(k, t1, t2):
    """Independent oracle: E = +/-sqrt(t1^2+t2^2+2*t1*t2*cos(k))."""
    _validate_hoppings(t1, t2)
    k = np.asarray(k, dtype=float)
    radicand = t1**2 + t2**2 + 2 * t1 * t2 * np.cos(k)
    # The radicand is nonnegative mathematically; protect against roundoff.
    positive = np.sqrt(np.maximum(radicand, 0.0))
    return np.stack((-positive, positive), axis=-1)


def band_gap(bands):
    """Full valence-to-conduction gap, measured from the numerical spectrum."""
    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 2 or bands.shape[1] != 2 or len(bands) == 0:
        raise ValueError("bands must have shape (num_k, 2), with num_k > 0.")
    return float(np.min(bands[:, 1]) - np.max(bands[:, 0]))


def run(num_k, output_dir):
    """Save one three-panel figure, per-case CSV files and an execution summary."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = k_grid(num_k)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True,
                             layout="constrained")
    reports = []
    for axis, (t1, t2) in zip(axes, CASES):
        numeric = numerical_bands(k, t1, t2)
        analytic = analytic_bands(k, t1, t2)
        numerical_gap = band_gap(numeric)
        expected_gap = 2 * abs(t1 - t2)
        error = float(np.max(np.abs(numeric - analytic)))
        gap_error = abs(numerical_gap - expected_gap)
        data_file = f"bands_t1_{t1:.1f}_t2_{t2:.1f}.csv"
        with (output_dir / data_file).open("x", encoding="utf-8") as stream:
            np.savetxt(stream, np.column_stack((k, numeric, analytic)),
                       delimiter=",", fmt="%.17g", comments="",
                       header="k,E_lower_numeric,E_upper_numeric,E_lower_analytic,E_upper_analytic")
        for band in range(2):
            axis.plot(k, numeric[:, band], color="#1764ab", linewidth=2,
                      label="NumPy eigh" if band == 0 else None)
            axis.plot(k, analytic[:, band], color="#d87518", linestyle="--",
                      linewidth=1.1, label="Analytic" if band == 0 else None)
        axis.axhline(0, color="#8b8b8b", linewidth=0.7, zorder=0)
        axis.set(title=rf"$t_1={t1:.1f},\ t_2={t2:.1f}$", xlabel=r"$k$ ($a=1$)",
                 xlim=(-np.pi, np.pi), ylim=(-2.2, 2.2),
                 xticks=(-np.pi, 0, np.pi), xticklabels=(r"$-\pi$", "0", r"$\pi$"))
        axis.grid(alpha=0.18)
        axis.text(0.5, 0.5, f"Full gap: {numerical_gap:.4g}",
                  transform=axis.transAxes, ha="center", va="center", fontsize=10,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        reports.append({
            "t1": t1, "t2": t2,
            "numerical_gap": numerical_gap,
            "analytic_gap": expected_gap,
            "gap_abs_error": gap_error,
            "max_abs_energy_error": error,
            "energy_check_passed": error <= ATOL,
            "gap_check_passed": gap_error <= ATOL,
            "csv": data_file,
        })
    axes[0].set_ylabel("Energy (hopping units)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False, fontsize=9)
    fig.suptitle("SSH bulk bands: numerical eigenvalues vs analytic spectrum", fontsize=13)
    fig.savefig(output_dir / "ssh_bands.png", dpi=180)
    plt.close(fig)
    summary = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()), "python": sys.version,
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "lattice_constant": 1.0,
        "k_min": float(k[0]), "k_max": float(k[-1]),
        "num_k": len(k), "both_zone_edges_included": True,
        "eigensolver": "numpy.linalg.eigh",
        "band_order": ["lower", "upper"],
        "absolute_tolerance": ATOL, "relative_tolerance": 0.0,
        "output_dir": str(output_dir), "plot": "ssh_bands.png",
        "cases": reports,
        "all_checks_passed": all(r["energy_check_passed"] and r["gap_check_passed"]
                                 for r in reports),
        "notes": [
            "Gap is min(E_upper)-max(E_lower), evaluated on the numerical grid.",
            "For positive hoppings the gap minimum lies at a zone edge included in this grid.",
            "A near-zero gap at equal hoppings is judged using the stated floating-point tolerance.",
            "Matching spectra for swapped t1/t2 do not independently establish topology.",
        ],
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-k", type=int, default=1001)
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; existing directories are preserved.")
    args = parser.parse_args()
    # Validate before creating any output files.
    k_grid(args.num_k)
    output_dir = args.output_dir or PROJECT_ROOT / "results" / datetime.now(timezone.utc).strftime(
        "ssh_bands_%Y%m%dT%H%M%S_%fZ")
    summary = run(args.num_k, output_dir)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
