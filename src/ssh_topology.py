"""SSH winding and open-chain states, reusing the existing bulk convention.

Run from the project root: python -B -m src.ssh_topology
No changes to ssh_bands.py are needed. Energies use the input hopping unit.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

if __package__:
    from .ssh_bands import CASES, PROJECT_ROOT, bloch_hamiltonian, k_grid
else:
    from ssh_bands import CASES, PROJECT_ROOT, bloch_hamiltonian, k_grid


GAP_RTOL = 1e-12
CHECK_ATOL = 1e-12
EDGE_CELLS = 3
EDGE_WEIGHT_MIN = 0.9
NEAR_ZERO_FRACTION = 0.01  # Relative to the bulk half-gap, not machine zero.
SIZES = (4, 8, 12, 16, 20, 24)


def q_path(k, t1, t2):
    """Upper-right Bloch element: q=dx-i*dy=t1+t2*exp(-i*k)."""
    return bloch_hamiltonian(k, t1, t2)[..., 0, 1]


def winding_number(t1, t2, num_k=2001, gap_rtol=GAP_RTOL):
    """Measure phase increments; nu=-w_q matches notes/ssh_theory.md (T9).

    Only gapped, resolved curves receive an integer. For this known circle,
    min|q|=|t1-t2| also detects a zero missed by a momentum mesh. The bound
    |d arg(q)/dk| <= t2/min|q| prevents undetected phase aliasing per interval.
    The hopping inequality is used for resolution checks, never to assign nu.
    """
    if not np.isfinite(gap_rtol) or gap_rtol <= 0:
        raise ValueError("gap_rtol must be finite and positive.")
    k = k_grid(num_k)
    q = q_path(k, t1, t2)
    distance = abs(t1 - t2)
    record = {
        "t1": t1, "t2": t2, "num_k": num_k,
        "sampled_min_abs_q": float(np.min(np.abs(q))),
        "circle_min_abs_q": float(distance),
        "gap_rtol": gap_rtol, "gap_threshold": gap_rtol * (t1 + t2),
        "status": "gap_closed", "w_q_raw": None, "w_q": None,
        "nu_raw": None, "nu": None, "integer_error": None,
        "phase_step_bound": None,
    }
    if t1 == t2:
        return record
    if distance <= record["gap_threshold"]:
        record["status"] = "gap_unresolved"
        return record
    bound = float(np.max(np.diff(k)) * t2 / distance)
    record["phase_step_bound"] = bound
    if bound >= np.pi / 2:
        record["status"] = "grid_unresolved"
        return record
    unit_q = q / np.abs(q)
    increments = np.angle(unit_q[1:] * unit_q[:-1].conjugate())
    w_raw = float(np.sum(increments) / (2 * np.pi))
    nu_raw = -w_raw
    nu = int(np.rint(nu_raw))
    integer_error = abs(nu_raw - nu)
    if integer_error > 1e-8:
        raise ArithmeticError("Phase sum did not resolve an integer winding.")
    record.update(status="defined", w_q_raw=w_raw, w_q=-nu,
                  nu_raw=nu_raw, nu=nu, integer_error=integer_error)
    return record


def open_chain_hamiltonian(num_cells, t1, t2):
    """Basis (A1,B1,...,AN,BN); endpoints A1 and BN, no wraparound bond.

    Nonnegative hoppings are allowed here to test the exact dimer limits;
    winding calculations retain the bulk module's strictly positive domain.
    """
    if (isinstance(num_cells, (bool, np.bool_)) or
            not isinstance(num_cells, (int, np.integer)) or num_cells < 1):
        raise ValueError("num_cells must be an integer >= 1.")
    if not all(np.isreal(t) and np.isfinite(t) and t >= 0 for t in (t1, t2)):
        raise ValueError("Open-chain hoppings must be finite nonnegative reals.")
    h = np.zeros((2 * num_cells, 2 * num_cells), dtype=float)
    bonds = np.empty(2 * num_cells - 1)
    bonds[0::2] = t1
    bonds[1::2] = t2
    i = np.arange(2 * num_cells - 1)
    h[i, i + 1] = bonds
    h[i + 1, i] = bonds
    return h


def chain_solution(num_cells, t1, t2, edge_cells=EDGE_CELLS):
    """Return all eigenvectors plus measured edge weights and solver checks."""
    h = open_chain_hamiltonian(num_cells, t1, t2)
    if (isinstance(edge_cells, (bool, np.bool_)) or
            not isinstance(edge_cells, (int, np.integer)) or
            edge_cells < 1 or 2 * edge_cells > num_cells):
        raise ValueError("edge_cells must be positive with nonoverlapping ends.")
    energies, states = np.linalg.eigh(h)
    probability = np.abs(states)**2  # Site x eigenstate; columns normalized.
    cells = probability.reshape(num_cells, 2, 2 * num_cells).sum(axis=1)
    left = cells[:edge_cells].sum(axis=0)
    right = cells[-edge_cells:].sum(axis=0)
    # E=0 is the chiral center; threshold refers to the infinite bulk half-gap.
    near_zero_threshold = NEAR_ZERO_FRACTION * abs(t1 - t2)
    edge_mask = ((np.abs(energies) < near_zero_threshold) &
                 (left + right >= EDGE_WEIGHT_MIN))
    central = np.argsort(np.abs(energies), kind="stable")[:2]
    central = central[np.argsort(energies[central])]
    record = {
        "num_cells": num_cells, "num_sites": 2 * num_cells,
        "t1": t1, "t2": t2, "edge_cells_each_end": edge_cells,
        "edge_weight_min": EDGE_WEIGHT_MIN,
        "near_zero_threshold": near_zero_threshold,
        "edge_state_indices": np.flatnonzero(edge_mask).tolist(),
        "central_indices": central.tolist(),
        "central_energies": energies[central].tolist(),
        "central_left_weights": left[central].tolist(),
        "central_right_weights": right[central].tolist(),
        "central_edge_weights": (left + right)[central].tolist(),
        "central_splitting": float(np.diff(energies[central])[0]),
        "hermiticity_error": float(np.max(np.abs(h - h.T.conjugate()))),
        "eigenpair_max_residual": float(np.max(np.abs(h @ states - states * energies))),
        "orthonormality_error": float(np.max(np.abs(states.T.conjugate() @ states - np.eye(len(h))))),
        "spectral_pairing_error": float(np.max(np.abs(energies + energies[::-1]))),
    }
    return energies, states, record


def _csv(path, data, header):
    with path.open("x", encoding="utf-8") as stream:
        np.savetxt(stream, data, delimiter=",", header=header,
                   comments="", fmt="%.17g")


def run(num_k, num_cells, output_dir):
    """Generate trajectories, all open-chain states, figures and JSON evidence."""
    k = k_grid(num_k)
    winding = [winding_number(t1, t2, num_k) for t1, t2 in CASES]
    if num_cells < 2 * EDGE_CELLS:
        raise ValueError(f"Use at least {2 * EDGE_CELLS} cells for edge windows.")
    chains = [chain_solution(num_cells, t1, t2) for t1, t2 in CASES[:2]]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    for ax, result in zip(axes, winding):
        t1, t2 = result["t1"], result["t2"]
        q = q_path(k, t1, t2)
        stem = f"t1_{t1:.1f}_t2_{t2:.1f}"
        _csv(output_dir / f"q_{stem}.csv", np.column_stack((k, q.real, q.imag, np.abs(q))),
             "k,q_real,q_imag,abs_q")
        ax.plot(q.real, q.imag, lw=2)
        ax.scatter([0], [0], c="black", marker="+", s=70, label="Origin", zorder=5)
        # Arrow on the upper half of the circle for increasing k.
        ka, kb = -0.60 * np.pi, -0.45 * np.pi
        qa, qb = q_path(np.array([ka, kb]), t1, t2)
        ax.annotate("", xy=(qb.real, qb.imag), xytext=(qa.real, qa.imag),
                    arrowprops={"arrowstyle": "->", "color": "#d87518", "lw": 2})
        if result["status"] == "defined":
            label = f"w(q)={result['w_q']}, SSH nu={result['nu']}"
        else:
            label = f"Undefined: {result['status']}"
        ax.set(title=f"t1={t1:g}, t2={t2:g}\n{label}", xlabel="Re q", ylabel="Im q",
               xlim=(-1.3, 2.3), ylim=(-1.8, 1.8), aspect="equal")
        ax.grid(alpha=0.2)
    fig.suptitle("SSH complex trajectories: arrows follow increasing k (-pi to pi)")
    fig.savefig(output_dir / "q_trajectories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
    chain_records = []
    for row, ((t1, t2), (energies, states, record)) in enumerate(zip(CASES[:2], chains)):
        stem = f"t1_{t1:.1f}_t2_{t2:.1f}"
        probability = np.abs(states)**2
        edge_weights = probability[:2 * EDGE_CELLS].sum(axis=0) + probability[-2 * EDGE_CELLS:].sum(axis=0)
        _csv(output_dir / f"spectrum_{stem}.csv",
             np.column_stack((np.arange(len(energies)), energies, edge_weights)),
             "state_index_zero_based,energy,edge_weight")
        # Real Hamiltonian: real normalized eigenvectors, one column per state.
        _csv(output_dir / f"states_{stem}.csv", states,
             ",".join(f"psi_state_{j}" for j in range(len(energies))))
        _csv(output_dir / f"probabilities_{stem}.csv", probability,
             ",".join(f"P_state_{j}" for j in range(len(energies))))
        center = record["central_indices"]
        ax = axes[row, 0]
        ax.scatter(np.arange(len(energies)), energies, s=19, color="#1764ab")
        ax.scatter(center, energies[center], s=40, color="#d87518", zorder=3,
                   label="Two closest to zero")
        ax.axhspan(-abs(t1 - t2), abs(t1 - t2), color="gray", alpha=0.12,
                   label="Infinite bulk gap")
        ax.axhline(0, lw=0.7, color="black")
        ax.set(xlabel="Eigenstate index (0 based)", ylabel="Energy",
               title=f"N={num_cells}, t1={t1:g}, t2={t2:g}\n{len(record['edge_state_indices'])} near-zero edge states")
        ax.legend(fontsize=8)
        sites = np.arange(1, 2 * num_cells + 1)
        for j, linestyle in zip(center, ("-", "--")):
            axes[row, 1].plot(sites, probability[:, j], linestyle=linestyle,
                              label=f"j={j}, E={energies[j]:+.4e}", lw=1.8)
        axes[row, 1].set(title="Actual eigenstates closest to zero", xlabel="Site (A1,B1,...)",
                         ylabel="Site probability", xlim=(1, len(sites)))
        axes[row, 1].legend(fontsize=8)
        # An extended state away from the gap is a useful localization control.
        bulk_index = num_cells // 2
        axes[row, 2].plot(sites, probability[:, bulk_index], color="#308050")
        axes[row, 2].set(title=f"Bulk comparison: j={bulk_index}, E={energies[bulk_index]:+.3f}",
                         xlabel="Site (A1,B1,...)", ylabel="Site probability", xlim=(1, len(sites)))
        for ax in axes[row]:
            ax.grid(alpha=0.18)
        record["bulk_representative_index"] = bulk_index
        chain_records.append(record)
    fig.suptitle("Open SSH chain: eigenvalues and normalized eigenstate probabilities")
    fig.savefig(output_dir / "open_chain_states.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    size_rows = []
    t1, t2 = CASES[1]
    ratio = t1 / t2
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for n in SIZES:
        h = open_chain_hamiltonian(n, t1, t2)
        energies = np.linalg.eigvalsh(h)
        em, ep = energies[n - 1:n + 1]
        # Overlap of normalized truncated semi-infinite end-state ansatzes.
        estimate = t2 * (1 - ratio**2) * ratio**n / (1 - ratio**(2 * n))
        residual = float(np.max(np.abs(energies + energies[::-1])))
        size_rows.append([n, em, ep, ep - em, 2 * estimate, residual])
    size_data = np.array(size_rows)
    _csv(output_dir / "finite_size.csv", size_data,
         "N,E_negative,E_positive,numerical_splitting,overlap_estimate_splitting,pairing_error")
    ax.semilogy(size_data[:, 0], size_data[:, 3], "o-", label="Numerical E+ - E-")
    ax.semilogy(size_data[:, 0], size_data[:, 4], "x--", label="End-state overlap estimate")
    ax.set(xlabel="Number of unit cells N", ylabel="Energy splitting",
           title="Finite size: t1=0.5, t2=1.5 (estimate is asymptotic)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_dir / "finite_size_splitting.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    checks = {
        "winding_matches_theory": all(r["status"] == "defined" and r["nu"] == expected
                                      for r, expected in zip(winding[:2], (0, 1))),
        "critical_winding_not_assigned": winding[2]["status"] == "gap_closed" and winding[2]["nu"] is None,
        "chain_solver_checks": all(max(r[key] for key in (
            "hermiticity_error", "eigenpair_max_residual", "orthonormality_error",
            "spectral_pairing_error")) <= CHECK_ATOL * max(1, r["t1"] + r["t2"])
                                   for r in chain_records),
        "trivial_has_no_near_zero_edge_states": len(chain_records[0]["edge_state_indices"]) == 0,
        "nontrivial_has_two_near_zero_edge_states": len(chain_records[1]["edge_state_indices"]) == 2,
        "splitting_positive_and_decreases": bool(np.all(size_data[:, 3] > 0) and
                                                np.all(np.diff(size_data[:, 3]) < 0)),
        "overlap_estimate_within_two_percent": bool(np.all(np.abs(size_data[:, 3] / size_data[:, 4] - 1) < 0.02)),
    }
    summary = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": subprocess.list2cmdline([sys.executable, *sys.orig_argv[1:]]),
        "cwd": str(Path.cwd()), "python": sys.version,
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "bulk_core_sha256": hashlib.sha256((PROJECT_ROOT / "src" / "ssh_bands.py").read_bytes()).hexdigest(),
        "convention": {"basis": "(A,B)", "q": "t1+t2*exp(-ik)",
                       "k_direction": "-pi to +pi", "w_q": "Delta arg(q)/(2*pi)",
                       "nu": "-w_q = Delta arg(q*)/(2*pi)",
                       "open_basis": "(A1,B1,...,AN,BN)", "termination": "A1 and BN; cut t2 bond",
                       "lattice_constant": 1, "energy_unit": "input hopping unit"},
        "winding": winding, "open_chains": chain_records,
        "finite_size": {"t1": t1, "t2": t2, "sizes": list(SIZES),
                        "columns": ["N", "E_negative", "E_positive", "splitting", "estimate", "pairing_error"],
                        "rows": size_rows},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "limitations": [
            "Critical winding is undefined; near-zero gaps or coarse meshes return unresolved, not an integer.",
            "Near-zero edge classification requires BOTH the stated energy and localization thresholds.",
            "Finite-chain hybridized eigenstates can have probability at BOTH ends; single-end combinations are not exact eigenstates.",
            "Overlap estimate is asymptotic; exponentially small splittings eventually fall below floating-point resolution.",
            "This is a clean noninteracting chiral SSH calculation for the stated termination, not a disorder/perturbation robustness study.",
        ],
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-k", type=int, default=2001)
    parser.add_argument("--num-cells", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or PROJECT_ROOT / "results" / datetime.now(timezone.utc).strftime(
        "ssh_topology_%Y%m%dT%H%M%S_%fZ")
    summary = run(args.num_k, args.num_cells, output)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
