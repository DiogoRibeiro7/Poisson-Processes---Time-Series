"""Tests for periodic Poisson simulation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import ArrayLike

from poisson_time_series import PoissonSimulation, simulate_periodic_poisson


def test_simulation_is_reproducible_with_explicit_generator() -> None:
    first = simulate_periodic_poisson(
        [2.0, 5.0, 10.0],
        cycles=4,
        rng=np.random.default_rng(42),
    )
    second = simulate_periodic_poisson(
        [2.0, 5.0, 10.0],
        cycles=4,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_array_equal(first.intensity, second.intensity)


def test_simulation_repeats_intensity_profile_exactly() -> None:
    simulation = simulate_periodic_poisson(
        [1.0, 3.0, 7.0],
        cycles=2,
        rng=np.random.default_rng(1),
    )

    np.testing.assert_allclose(simulation.intensity, [1.0, 3.0, 7.0, 1.0, 3.0, 7.0])
    assert simulation.period == 3
    assert simulation.cycles == 2
    assert simulation.nobs == 6
    assert np.issubdtype(simulation.counts.dtype, np.integer)
    assert not simulation.counts.flags.writeable
    assert not simulation.intensity.flags.writeable


def test_zero_intensity_produces_zero_counts() -> None:
    simulation = simulate_periodic_poisson(
        [0.0, 0.0],
        cycles=3,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_array_equal(simulation.counts, np.zeros(6, dtype=np.int64))


@pytest.mark.parametrize(
    "rates",
    [
        [],
        [1.0, np.nan],
        [1.0, np.inf],
        [-1.0, 2.0],
        [[1.0, 2.0]],
    ],
)
def test_simulation_rejects_invalid_rates(rates: ArrayLike) -> None:
    with pytest.raises(ValueError):
        simulate_periodic_poisson(rates)


def test_simulation_rejects_non_numeric_rates() -> None:
    with pytest.raises(TypeError, match="numeric"):
        simulate_periodic_poisson(["a", "b"])


@pytest.mark.parametrize("cycles", [0, -1])
def test_simulation_rejects_invalid_cycle_range(cycles: int) -> None:
    with pytest.raises(ValueError):
        simulate_periodic_poisson([1.0, 2.0], cycles=cycles)


def test_simulation_rejects_non_integer_cycles() -> None:
    with pytest.raises(TypeError):
        simulate_periodic_poisson([1.0, 2.0], cycles=cast(int, 1.5))


def test_simulation_rejects_invalid_rng() -> None:
    with pytest.raises(TypeError, match="Generator"):
        simulate_periodic_poisson(
            [1.0, 2.0],
            rng=cast(np.random.Generator, object()),
        )


@pytest.mark.parametrize(
    ("counts", "intensity", "period"),
    [
        (np.array([[1, 2]]), np.array([1.0, 2.0]), 2),
        (np.array([1, 2]), np.array([[1.0, 2.0]]), 2),
        (np.array([], dtype=np.int64), np.array([], dtype=np.float64), 1),
        (np.array([1, 2]), np.array([1.0]), 1),
        (np.array([-1, 2]), np.array([1.0, 2.0]), 2),
        (np.array([1, 2]), np.array([1.0, np.nan]), 2),
        (np.array([1, 2]), np.array([1.0, 2.0]), 0),
        (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), 2),
    ],
)
def test_simulation_result_validates_invariants(
    counts: np.ndarray,
    intensity: np.ndarray,
    period: int,
) -> None:
    with pytest.raises(ValueError):
        PoissonSimulation(counts=counts, intensity=intensity, period=period)
