"""Tests for fixed-dispersion NB2 INGARCH(1,1) models."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import (
    fit_negative_binomial_ingarch,
    simulate_negative_binomial_ingarch,
)


def test_nb_ingarch_simulation_is_reproducible() -> None:
    first = simulate_negative_binomial_ingarch(
        120,
        omega=1.0,
        alpha=0.20,
        beta=0.55,
        dispersion=0.30,
        rng=np.random.default_rng(42),
    )
    second = simulate_negative_binomial_ingarch(
        120,
        omega=1.0,
        alpha=0.20,
        beta=0.55,
        dispersion=0.30,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_allclose(first.intensity, second.intensity)
    np.testing.assert_allclose(first.conditional_variance, second.conditional_variance)
    assert not first.counts.flags.writeable
    assert not first.intensity.flags.writeable
    assert not first.conditional_variance.flags.writeable


def test_nb_ingarch_simulation_obeys_mean_and_variance_recursions() -> None:
    simulation = simulate_negative_binomial_ingarch(
        80,
        omega=0.8,
        alpha=0.20,
        beta=0.60,
        dispersion=0.40,
        burn_in=100,
        rng=np.random.default_rng(7),
    )

    expected_mean = (
        0.8
        + 0.20 * simulation.counts[:-1]
        + 0.60 * simulation.intensity[:-1]
    )
    np.testing.assert_allclose(simulation.intensity[1:], expected_mean)
    np.testing.assert_allclose(
        simulation.conditional_variance,
        simulation.intensity + 0.40 * simulation.intensity**2,
    )


def test_nb_ingarch_unconditional_mean_matches_parameter_formula() -> None:
    simulation = simulate_negative_binomial_ingarch(
        8000,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        dispersion=0.25,
        burn_in=800,
        rng=np.random.default_rng(123),
    )

    expected_mean = 1.0 / (1.0 - 0.20 - 0.50)
    assert float(np.mean(simulation.counts)) == pytest.approx(expected_mean, abs=0.30)


def test_nb_ingarch_fit_recovers_temporal_structure() -> None:
    simulation = simulate_negative_binomial_ingarch(
        2200,
        omega=1.0,
        alpha=0.20,
        beta=0.55,
        dispersion=0.30,
        burn_in=700,
        rng=np.random.default_rng(21),
    )

    result = fit_negative_binomial_ingarch(
        simulation.counts,
        dispersion=0.30,
    )

    true_mean = 1.0 / (1.0 - 0.20 - 0.55)
    true_persistence = 0.20 + 0.55

    assert result.converged
    assert result.unconditional_mean == pytest.approx(true_mean, abs=0.40)
    assert result.alpha == pytest.approx(0.20, abs=0.10)
    assert result.alpha + result.beta == pytest.approx(true_persistence, abs=0.12)
    assert result.dispersion == pytest.approx(0.30)
    assert result.alpha + result.beta < 1.0
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)


def test_nb_ingarch_fit_retains_conditional_variance() -> None:
    simulation = simulate_negative_binomial_ingarch(
        300,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        dispersion=0.35,
        rng=np.random.default_rng(4),
    )
    result = fit_negative_binomial_ingarch(
        simulation.counts,
        dispersion=0.35,
    )

    np.testing.assert_allclose(
        result.fitted_conditional_variance,
        result.fitted_intensity + 0.35 * result.fitted_intensity**2,
    )
    assert not result.observed_counts.flags.writeable
    assert not result.fitted_intensity.flags.writeable
    assert not result.fitted_conditional_variance.flags.writeable


def test_nb_ingarch_forecast_mean_and_one_step_variance() -> None:
    simulation = simulate_negative_binomial_ingarch(
        400,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        dispersion=0.25,
        rng=np.random.default_rng(9),
    )
    result = fit_negative_binomial_ingarch(
        simulation.counts,
        dispersion=0.25,
    )

    forecast = result.forecast_mean(3)
    first = (
        result.omega
        + result.alpha * result.observed_counts[-1]
        + result.beta * result.fitted_intensity[-1]
    )
    second = result.omega + (result.alpha + result.beta) * first

    assert forecast[0] == pytest.approx(first)
    assert forecast[1] == pytest.approx(second)
    assert result.one_step_forecast_variance() == pytest.approx(
        first + result.dispersion * first**2
    )


@pytest.mark.parametrize("dispersion", [0.0, -1.0, np.inf, np.nan])
def test_nb_ingarch_rejects_invalid_dispersion(dispersion: float) -> None:
    with pytest.raises(ValueError):
        simulate_negative_binomial_ingarch(
            10,
            omega=1.0,
            alpha=0.20,
            beta=0.50,
            dispersion=dispersion,
        )


def test_nb_ingarch_rejects_non_numeric_dispersion() -> None:
    with pytest.raises(TypeError):
        fit_negative_binomial_ingarch(
            [1, 2, 3],
            dispersion=cast(float, "bad"),
        )


@pytest.mark.parametrize(
    ("omega", "alpha", "beta"),
    [
        (0.0, 0.2, 0.5),
        (1.0, -0.1, 0.5),
        (1.0, 0.2, -0.1),
        (1.0, 0.6, 0.4),
    ],
)
def test_nb_ingarch_rejects_invalid_dynamics(
    omega: float,
    alpha: float,
    beta: float,
) -> None:
    with pytest.raises(ValueError):
        simulate_negative_binomial_ingarch(
            10,
            omega=omega,
            alpha=alpha,
            beta=beta,
            dispersion=0.30,
        )


def test_nb_ingarch_fit_requires_enough_counts() -> None:
    with pytest.raises(ValueError, match="at least three"):
        fit_negative_binomial_ingarch([1, 2], dispersion=0.30)


def test_nb_ingarch_forecast_rejects_invalid_horizon() -> None:
    simulation = simulate_negative_binomial_ingarch(
        100,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        dispersion=0.25,
        rng=np.random.default_rng(10),
    )
    result = fit_negative_binomial_ingarch(
        simulation.counts,
        dispersion=0.25,
    )

    with pytest.raises(ValueError):
        result.forecast_mean(0)

    with pytest.raises(TypeError):
        result.forecast_mean(cast(int, 1.5))
