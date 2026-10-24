"""Physical checks for the requested positive-hopping SSH benchmark cases."""

import numpy as np
import pytest

from src.ssh_bands import (
    analytic_bands,
    band_gap,
    bloch_hamiltonian,
    k_grid,
    numerical_bands,
)


CASES = [(1.5, 0.5), (0.5, 1.5), (1.0, 1.0)]
ATOL = 1e-12


@pytest.mark.parametrize("t1,t2", CASES)
def test_hamiltonian_is_hermitian(t1, t2):
    matrices = bloch_hamiltonian(k_grid(), t1, t2)
    np.testing.assert_allclose(matrices, matrices.conj().swapaxes(-1, -2),
                               atol=ATOL, rtol=0)


def test_hopping_phase_at_quarter_turn():
    # Independent fixed-momentum matrix check detects reversed exp(+/-ik) signs.
    expected = np.array([[0, 1.5 - 0.5j], [1.5 + 0.5j, 0]])
    np.testing.assert_allclose(bloch_hamiltonian(np.pi / 2, 1.5, 0.5),
                               expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("t1,t2", CASES)
def test_eigh_matches_independent_analytic_spectrum(t1, t2):
    k = k_grid()
    numeric = numerical_bands(k, t1, t2)
    np.testing.assert_allclose(numeric, analytic_bands(k, t1, t2),
                               atol=ATOL, rtol=0)
    assert np.all(numeric[:, 0] <= numeric[:, 1])


@pytest.mark.parametrize("t1,t2,expected_gap", [(1.5, 0.5, 2.0),
                                             (0.5, 1.5, 2.0),
                                             (1.0, 1.0, 0.0)])
def test_full_gap_from_numerical_spectrum(t1, t2, expected_gap):
    assert band_gap(numerical_bands(k_grid(), t1, t2)) == pytest.approx(
        expected_gap, abs=ATOL, rel=0)
    # Unequal positive hoppings must remain gapped.
    if t1 != t2:
        assert expected_gap > ATOL


@pytest.mark.parametrize("hopping", [0.4, 1.0, 2.0])
def test_equal_hoppings_close_only_at_zone_edges(hopping):
    edges = numerical_bands(np.array([-np.pi, np.pi]), hopping, hopping)
    np.testing.assert_allclose(edges, 0.0, atol=ATOL, rtol=0)
    # A critical chain is not the zero Hamiltonian: k=0 remains away from zero.
    np.testing.assert_allclose(numerical_bands(0.0, hopping, hopping),
                               [-2 * hopping, 2 * hopping], atol=ATOL, rtol=0)
    interior = numerical_bands(np.linspace(-np.pi + 0.01, np.pi - 0.01, 251),
                               hopping, hopping)
    assert np.all(interior[:, 1] > ATOL)


@pytest.mark.parametrize("t1,t2", CASES)
def test_brillouin_zone_periodicity(t1, t2):
    k = np.array([-2.7, -0.3, 0.0, 1.2, 2.9])
    np.testing.assert_allclose(bloch_hamiltonian(k + 2 * np.pi, t1, t2),
                               bloch_hamiltonian(k, t1, t2), atol=ATOL, rtol=0)
