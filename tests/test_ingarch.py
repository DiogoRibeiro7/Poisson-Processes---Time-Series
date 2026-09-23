"""Tests for Poisson INGARCH(1,1) models."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import fit_poisson_ingarch, simulate_poisson_ingarch


def test_ingarch_simulation_is_reproducible() -> None:
    first = simulate_poisson_ingarch(
        100,
        omega=1.2,
        alpha=0.25,
        beta=0.50,
        rng=np.random.default_rng(42),
    )
    second = simulate_poisson_ingarch(
        100,
        omega=1.2,
        alpha=0.25,
        beta=0.50,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_allclose(first.intensity, second.intensity)
    assert not first.counts.flags.writeable
    assert not first.intensity.flags.writeable


def test_ingarch_simulation_obeys_recursion_after_retained_start() -> None:
    simulation = simulate_poisson_ingarch(
        50,
        omega=0.8,
        alpha=0.20,
        beta=0.60,
        burn_in=50,
        rng=np.random.default_rng(7),
    )

    expected = (
        0.8
        + 0.20 * simulation.counts[:-1]
        + 0.60 * simulation.intensity[:-1]
    )
    np.testing.assert_allclose(simulation.intensity[1:], expected)


def test_ingarch_unconditional_mean_matches_parameter_formula() -> None:
    simulation = simulate_poisson_ingarch(
        6000,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        burn_in=500,
        rng=np.random.default_rng(123),
    )

    expected_mean = 1.0 / (1.0 - 0.20 - 0.50)
    assert float(np.mean(simulation.counts)) == pytest.approx(expected_mean, abs=0.20)


def test_ingarch_fit_recovers_simulated_parameters() -> None:
    simulation = simulate_poisson_ingarch(
        1500,
        omega=1.2,
        alpha=0.25,
        beta=0.50,
        burn_in=500,
        rng=np.random.default_rng(42),
    )

    result = fit_poisson_ingarch(simulation.counts)

    assert result.converged
    assert result.omega == pytest.approx(1.2, abs=0.25)
    assert result.alpha == pytest.approx(0.25, abs=0.08)
    assert result.beta == pytest.approx(0.50, abs=0.10)
    assert result.alpha + result.beta < 1.0
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert not result.observed_counts.flags.writeable
    assert not result.fitted_intensity.flags.writeable


def test_ingarch_forecast_uses_observed_count_then_expected_recursion() -> None:
    simulation = simulate_poisson_ingarch(
        300,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        burn_in=100,
        rng=np.random.default_rng(4),
    )
    result = fit_poisson_ingarch(simulation.counts)

    forecasts = result.forecast_mean(3)
    first = (
        result.omega
        + result.alpha * result.observed_counts[-1]
        + result.beta * result.fitted_intensity[-1]
    )
    second = result.omega + (result.alpha + result.beta) * first

    assert forecasts[0] == pytest.approx(first)
    assert forecasts[1] == pytest.approx(second)
    assert not forecasts.flags.writeable


@pytest.mark.parametrize(
    ("omega", "alpha", "beta"),
    [
        (0.0, 0.2, 0.5),
        (-1.0, 0.2, 0.5),
        (1.0, -0.1, 0.5),
        (1.0, 0.2, -0.1),
        (1.0, 0.6, 0.4),
        (1.0, 0.7, 0.5),
    ],
)
def test_ingarch_simulation_rejects_invalid_dynamics(
    omega: float,
    alpha: float,
    beta: float,
) -> None:
    with pytest.raises(ValueError):
        simulate_poisson_ingarch(
            10,
            omega=omega,
            alpha=alpha,
            beta=beta,
        )


def test_ingarch_simulation_rejects_non_numeric_dynamics() -> None:
    with pytest.raises(TypeError):
        simulate_poisson_ingarch(
            10,
            omega=cast(float, "bad"),
            alpha=0.2,
            beta=0.5,
        )


@pytest.mark.parametrize("nobs", [0, -1])
def test_ingarch_simulation_rejects_invalid_nobs(nobs: int) -> None:
    with pytest.raises(ValueError):
        simulate_poisson_ingarch(
            nobs,
            omega=1.0,
            alpha=0.2,
            beta=0.5,
        )


def test_ingarch_simulation_rejects_non_integer_nobs() -> None:
    with pytest.raises(TypeError):
        simulate_poisson_ingarch(
            cast(int, 10.5),
            omega=1.0,
            alpha=0.2,
            beta=0.5,
        )


def test_ingarch_fit_requires_enough_counts() -> None:
    with pytest.raises(ValueError, match="at least three"):
        fit_poisson_ingarch([1, 2])


def test_ingarch_fit_rejects_invalid_initial_intensity() -> None:
    with pytest.raises(ValueError, match="initial_intensity"):
        fit_poisson_ingarch([1, 2, 3], initial_intensity=0.0)


def test_ingarch_fit_rejects_invalid_optimizer_controls() -> None:
    with pytest.raises(ValueError):
        fit_poisson_ingarch([1, 2, 3], max_iterations=0)

    with pytest.raises(ValueError):
        fit_poisson_ingarch([1, 2, 3], tolerance=0.0)


def test_ingarch_forecast_rejects_invalid_horizon() -> None:
    simulation = simulate_poisson_ingarch(
        100,
        omega=1.0,
        alpha=0.2,
        beta=0.5,
        rng=np.random.default_rng(10),
    )
    result = fit_poisson_ingarch(simulation.counts)

    with pytest.raises(ValueError):
        result.forecast_mean(0)

    with pytest.raises(TypeError):
        result.forecast_mean(cast(int, 1.5))
