"""Independent physical oracles for SSH topology and open-chain boundaries."""

import json

import numpy as np
import pytest

from src.ssh_bands import bloch_hamiltonian
from src.ssh_topology import (
    chain_solution,
    open_chain_hamiltonian,
    q_path,
    run,
    winding_number,
)


@pytest.mark.parametrize("t1,t2,nu,wq", [(1.5, 0.5, 0, 0), (0.5, 1.5, 1, -1)])
@pytest.mark.parametrize("num_k", [101, 1001, 2001])
def test_gapped_winding_and_grid_stability(t1, t2, nu, wq, num_k):
    result = winding_number(t1, t2, num_k)
    assert result["status"] == "defined"
    assert result["nu_raw"] == pytest.approx(nu, abs=1e-12, rel=0)
    assert result["w_q_raw"] == pytest.approx(wq, abs=1e-12, rel=0)
    assert result["nu"] == nu and result["w_q"] == wq


def test_q_sign_and_circle_geometry():
    k = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    np.testing.assert_allclose(q_path(k, 0.5, 1.5),
                               [-1, 0.5 + 1.5j, 2, 0.5 - 1.5j, -1], atol=1e-12)
    np.testing.assert_allclose(np.abs(q_path(k, 0.5, 1.5) - 0.5), 1.5, atol=1e-12)


@pytest.mark.parametrize("num_k", [100, 101, 2001])
def test_gap_closure_never_assigned_half_or_integer_winding(num_k):
    # Even point counts omit k=0, but still include the gap-closing zone edge.
    result = winding_number(1, 1, num_k)
    assert result["status"] == "gap_closed"
    assert result["nu"] is None and result["w_q"] is None
    assert result["nu_raw"] is None and result["w_q_raw"] is None


def test_near_critical_and_coarse_grid_fail_closed():
    assert winding_number(1, 1 + 1e-13)["status"] == "gap_unresolved"
    result = winding_number(0.999, 1, 101)
    assert result["status"] == "grid_unresolved" and result["nu"] is None
    assert winding_number(0.999, 1, 10001)["nu"] == 1
    assert winding_number(1, 0.999, 10001)["nu"] == 0


@pytest.mark.parametrize("scale", [1e-8, 1, 1e8])
def test_winding_is_invariant_under_energy_rescaling(scale):
    assert winding_number(0.5 * scale, 1.5 * scale)["nu"] == 1
    assert winding_number(1.5 * scale, 0.5 * scale)["nu"] == 0
    assert winding_number(scale, scale)["nu"] is None


def test_open_chain_exact_bonds_and_termination():
    expected = np.array([[0, 2, 0, 0, 0, 0], [2, 0, 3, 0, 0, 0],
                         [0, 3, 0, 2, 0, 0], [0, 0, 2, 0, 3, 0],
                         [0, 0, 0, 3, 0, 2], [0, 0, 0, 0, 2, 0]])
    np.testing.assert_array_equal(open_chain_hamiltonian(3, 2, 3), expected)
    np.testing.assert_array_equal(open_chain_hamiltonian(1, 2, 3), [[0, 2], [2, 0]])


def test_real_space_fourier_projection_matches_existing_bloch_convention():
    n = 7
    periodic = open_chain_hamiltonian(n, 0.5, 1.5)
    periodic[0, -1] = periodic[-1, 0] = 1.5
    for k in 2 * np.pi * np.arange(n) / n:
        plane_wave = np.zeros((2 * n, 2), dtype=complex)
        plane_wave[0::2, 0] = np.exp(1j * k * np.arange(n)) / np.sqrt(n)
        plane_wave[1::2, 1] = np.exp(1j * k * np.arange(n)) / np.sqrt(n)
        np.testing.assert_allclose(plane_wave.conj().T @ periodic @ plane_wave,
                                   bloch_hamiltonian(k, 0.5, 1.5), atol=1e-12, rtol=0)


@pytest.mark.parametrize("t1,t2", [(1.5, 0.5), (0.5, 1.5), (1, 1)])
def test_chiral_symmetry_eigenpairs_and_normalization(t1, t2):
    h = open_chain_hamiltonian(20, t1, t2)
    gamma = np.diag(np.tile([1, -1], 20))
    np.testing.assert_array_equal(h, h.T.conj())
    np.testing.assert_array_equal(gamma @ h @ gamma, -h)
    energies, states, record = chain_solution(20, t1, t2)
    np.testing.assert_allclose(h @ states, states * energies, atol=1e-12, rtol=0)
    np.testing.assert_allclose(states.conj().T @ states, np.eye(40), atol=1e-12, rtol=0)
    np.testing.assert_allclose(energies, -energies[::-1], atol=1e-12, rtol=0)
    assert record["eigenpair_max_residual"] < 1e-12


@pytest.mark.parametrize("t1,t2,expected_count", [(1.5, 0.5, 0), (0.5, 1.5, 2)])
def test_near_zero_and_localization_joint_criterion(t1, t2, expected_count):
    energies, states, record = chain_solution(20, t1, t2)
    assert len(record["edge_state_indices"]) == expected_count
    if expected_count:
        center = record["central_indices"]
        assert np.max(np.abs(energies[center])) < 1e-8
        assert min(record["central_edge_weights"]) > 0.99
        # Both hybridized eigenstates live at both ends for a resolved doublet.
        np.testing.assert_allclose(record["central_left_weights"], 0.5, atol=0.002)
        np.testing.assert_allclose(record["central_right_weights"], 0.5, atol=0.002)
        cells = np.abs(states[:, center]).reshape(20, 2, 2)**2
        cell_probability = cells.sum(axis=1)
        # Semi-infinite prediction: probability ratio = (t1/t2)^2.
        np.testing.assert_allclose(cell_probability[1:4] / cell_probability[:3],
                                   (t1 / t2)**2, atol=1e-8, rtol=0)
    else:
        assert np.min(np.abs(energies)) > abs(t1 - t2)


@pytest.mark.parametrize("n", [1, 4, 20])
def test_exact_dimer_limits(n):
    trivial = np.linalg.eigvalsh(open_chain_hamiltonian(n, 2, 0))
    np.testing.assert_allclose(trivial, [-2] * n + [2] * n, atol=1e-12)
    h = open_chain_hamiltonian(n, 0, 2)
    topological = np.linalg.eigvalsh(h)
    np.testing.assert_allclose(topological, [-2] * (n - 1) + [0, 0] + [2] * (n - 1), atol=1e-12)
    # Exact dangling sites in this termination, independent of eigensolver basis.
    assert np.all(h[:, 0] == 0) and np.all(h[:, -1] == 0)


@pytest.mark.parametrize("n", [3, 10, 20])
def test_uniform_open_chain_has_known_analytic_spectrum(n):
    numeric = np.linalg.eigvalsh(open_chain_hamiltonian(n, 1, 1))
    expected = np.sort(2 * np.cos(np.pi * np.arange(1, 2 * n + 1) / (2 * n + 1)))
    np.testing.assert_allclose(numeric, expected, atol=1e-12, rtol=0)
    assert np.min(np.abs(numeric)) > 0  # Finite even chain, despite gapless bulk.


def test_finite_size_splitting_and_overlap_estimate():
    sizes = np.array([4, 8, 12, 16, 20, 24])
    splitting = []
    for n in sizes:
        energy = np.linalg.eigvalsh(open_chain_hamiltonian(int(n), 0.5, 1.5))
        assert energy[n - 1] < 0 < energy[n]
        splitting.append(energy[n] - energy[n - 1])
    splitting = np.array(splitting)
    assert np.all(np.diff(splitting) < 0)
    r = 1 / 3
    estimate = 2 * 1.5 * (1 - r**2) * r**sizes / (1 - r**(2 * sizes))
    np.testing.assert_allclose(splitting, estimate, rtol=0.02, atol=0)


@pytest.mark.parametrize("n,t1,t2", [(0, 1, 1), (True, 1, 1), (2.5, 1, 1),
                                       (4, -1, 1), (4, np.nan, 1), (4, 1, np.inf)])
def test_invalid_chain_inputs(n, t1, t2):
    with pytest.raises(ValueError):
        open_chain_hamiltonian(n, t1, t2)


@pytest.mark.parametrize("t1,t2,num_k,tol", [(0, 1, 101, 1e-12), (1, -1, 101, 1e-12),
                                             (1, 2, 2, 1e-12), (1, 2, 101, 0)])
def test_invalid_winding_inputs(t1, t2, num_k, tol):
    with pytest.raises(ValueError):
        winding_number(t1, t2, num_k, tol)


def test_overlapping_edge_windows_rejected():
    with pytest.raises(ValueError, match="nonoverlapping"):
        chain_solution(4, 0.5, 1.5, edge_cells=3)


def test_saved_artifacts_reconstruct_the_eigenproblem(tmp_path):
    output = tmp_path / "new_run"
    result = run(101, 20, output)
    stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["all_checks_passed"] and stored["all_checks_passed"]
    assert stored["winding"][2]["nu"] is None
    for t1, t2 in [(1.5, 0.5), (0.5, 1.5)]:
        stem = f"t1_{t1:.1f}_t2_{t2:.1f}"
        states = np.loadtxt(output / f"states_{stem}.csv", delimiter=",", skiprows=1)
        probability = np.loadtxt(output / f"probabilities_{stem}.csv", delimiter=",", skiprows=1)
        spectrum = np.loadtxt(output / f"spectrum_{stem}.csv", delimiter=",", skiprows=1)
        np.testing.assert_allclose(open_chain_hamiltonian(20, t1, t2) @ states,
                                   states * spectrum[:, 1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(probability, np.abs(states)**2, atol=1e-15, rtol=0)
        np.testing.assert_allclose(probability.sum(axis=0), 1, atol=1e-12, rtol=0)
    for name in ["q_trajectories.png", "open_chain_states.png", "finite_size_splitting.png"]:
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(FileExistsError):
        run(101, 20, output)
