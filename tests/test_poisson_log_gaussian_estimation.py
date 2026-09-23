"""Tests for approximate Poisson log-Gaussian parameter estimation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import (
    fit_poisson_log_gaussian_parameters,
    laplace_log_marginal_likelihood,
    simulate_poisson_log_gaussian,
)


def _simulation() -> np.ndarray:
    simulation = simulate_poisson_log_gaussian(
        350,
        state_mean=0.8,
        phi=0.75,
        state_sd=0.25,
        burn_in=300,
        rng=np.random.default_rng(42),
    )
    return simulation.counts


def test_laplace_log_marginal_likelihood_is_finite() -> None:
    value = laplace_log_marginal_likelihood(
        _simulation(),
        state_mean=0.8,
        phi=0.75,
        state_sd=0.25,
    )
    assert np.isfinite(value)


def test_parameter_fit_improves_approximate_objective() -> None:
    counts = _simulation()
    initial_value = laplace_log_marginal_likelihood(
        counts,
        state_mean=0.0,
        phi=0.20,
        state_sd=0.60,
    )
    result = fit_poisson_log_gaussian_parameters(
        counts,
        initial_state_mean=0.0,
        initial_phi=0.20,
        initial_state_sd=0.60,
    )

    assert result.converged
    assert result.approximate_log_likelihood >= initial_value
    assert abs(result.phi) < 1.0
    assert result.state_sd > 0.0
    assert result.filter_result.converged
    assert result.nobs == counts.size
    assert result.nparams == 3
    assert np.isfinite(result.approximate_aic)
    assert np.isfinite(result.approximate_bic)


def test_parameter_fit_recovers_broad_latent_dynamics() -> None:
    result = fit_poisson_log_gaussian_parameters(_simulation())

    assert result.state_mean == pytest.approx(0.8, abs=0.45)
    assert result.phi == pytest.approx(0.75, abs=0.25)
    assert result.state_sd == pytest.approx(0.25, abs=0.20)


def test_parameter_fit_uses_default_state_mean_from_counts() -> None:
    counts = _simulation()
    result = fit_poisson_log_gaussian_parameters(counts)

    assert result.converged
    assert np.isfinite(result.state_mean)


def test_parameter_fit_requires_enough_observations() -> None:
    with pytest.raises(ValueError, match="at least five"):
        fit_poisson_log_gaussian_parameters([1, 2, 3, 4])


@pytest.mark.parametrize("initial_phi", [-1.0, 1.0, 1.5])
def test_parameter_fit_rejects_invalid_initial_phi(initial_phi: float) -> None:
    with pytest.raises(ValueError):
        fit_poisson_log_gaussian_parameters(
            [1, 2, 3, 4, 5],
            initial_phi=initial_phi,
        )


@pytest.mark.parametrize("initial_state_sd", [0.0, -0.1, np.nan, np.inf])
def test_parameter_fit_rejects_invalid_initial_state_sd(
    initial_state_sd: float,
) -> None:
    with pytest.raises(ValueError):
        fit_poisson_log_gaussian_parameters(
            [1, 2, 3, 4, 5],
            initial_state_sd=initial_state_sd,
        )


def test_parameter_fit_rejects_non_numeric_initial_state_mean() -> None:
    with pytest.raises(TypeError):
        fit_poisson_log_gaussian_parameters(
            [1, 2, 3, 4, 5],
            initial_state_mean=cast(float, "bad"),
        )


@pytest.mark.parametrize(
    ("optimizer_tolerance", "optimizer_max_iterations"),
    [(0.0, 100), (1e-8, 0)],
)
def test_parameter_fit_rejects_invalid_optimizer_controls(
    optimizer_tolerance: float,
    optimizer_max_iterations: int,
) -> None:
    with pytest.raises(ValueError):
        fit_poisson_log_gaussian_parameters(
            [1, 2, 3, 4, 5],
            optimizer_tolerance=optimizer_tolerance,
            optimizer_max_iterations=optimizer_max_iterations,
        )
