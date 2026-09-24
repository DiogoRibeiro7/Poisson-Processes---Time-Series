"""Tests for approximate log-Gaussian parameter uncertainty."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import (
    PoissonLogGaussianParameterFit,
    estimate_poisson_log_gaussian_uncertainty,
    fit_poisson_log_gaussian_parameters,
    simulate_poisson_log_gaussian,
)


@pytest.fixture(scope="module")
def fitted_model() -> PoissonLogGaussianParameterFit:
    simulation = simulate_poisson_log_gaussian(
        450,
        state_mean=0.8,
        phi=0.70,
        state_sd=0.25,
        burn_in=300,
        rng=np.random.default_rng(42),
    )
    return fit_poisson_log_gaussian_parameters(simulation.counts)


def test_uncertainty_returns_positive_standard_errors(
    fitted_model: PoissonLogGaussianParameterFit,
) -> None:
    result = estimate_poisson_log_gaussian_uncertainty(fitted_model)

    assert result.standard_errors.shape == (3,)
    assert np.all(result.standard_errors > 0.0)
    assert result.covariance_parameters.shape == (3, 3)
    assert result.covariance_unconstrained.shape == (3, 3)
    assert not result.standard_errors.flags.writeable


def test_uncertainty_covariances_are_symmetric(
    fitted_model: PoissonLogGaussianParameterFit,
) -> None:
    result = estimate_poisson_log_gaussian_uncertainty(fitted_model)

    np.testing.assert_allclose(
        result.covariance_parameters,
        result.covariance_parameters.T,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.covariance_unconstrained,
        result.covariance_unconstrained.T,
        atol=1e-10,
    )


def test_confidence_intervals_contain_point_estimates(
    fitted_model: PoissonLogGaussianParameterFit,
) -> None:
    result = estimate_poisson_log_gaussian_uncertainty(
        fitted_model,
        confidence_level=0.90,
    )
    estimates = np.array(
        [fitted_model.state_mean, fitted_model.phi, fitted_model.state_sd]
    )

    assert np.all(result.lower <= estimates)
    assert np.all(estimates <= result.upper)
    assert result.lower[1] >= -1.0
    assert result.upper[1] <= 1.0
    assert result.lower[2] >= 0.0


def test_parameter_correlation_has_unit_diagonal(
    fitted_model: PoissonLogGaussianParameterFit,
) -> None:
    result = estimate_poisson_log_gaussian_uncertainty(fitted_model)
    correlation = result.correlation_parameters

    np.testing.assert_allclose(np.diag(correlation), np.ones(3), atol=1e-10)
    assert not correlation.flags.writeable


def test_uncertainty_rejects_non_fit() -> None:
    with pytest.raises(TypeError, match="PoissonLogGaussianParameterFit"):
        estimate_poisson_log_gaussian_uncertainty(
            cast(PoissonLogGaussianParameterFit, object())
        )


@pytest.mark.parametrize("hessian_step", [0.0, -1.0, np.nan])
def test_uncertainty_rejects_invalid_hessian_step(
    hessian_step: float,
    fitted_model: PoissonLogGaussianParameterFit,
) -> None:
    with pytest.raises(ValueError):
        estimate_poisson_log_gaussian_uncertainty(
            fitted_model,
            hessian_step=hessian_step,
        )


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_uncertainty_rejects_invalid_confidence_level(
    confidence_level: float,
    fitted_model: PoissonLogGaussianParameterFit,
) -> None:
    with pytest.raises(ValueError):
        estimate_poisson_log_gaussian_uncertainty(
            fitted_model,
            confidence_level=confidence_level,
        )


def test_uncertainty_rejects_invalid_filter_iterations(
    fitted_model: PoissonLogGaussianParameterFit,
) -> None:
    with pytest.raises(ValueError):
        estimate_poisson_log_gaussian_uncertainty(
            fitted_model,
            filter_max_iterations=0,
        )

    with pytest.raises(TypeError):
        estimate_poisson_log_gaussian_uncertainty(
            fitted_model,
            filter_max_iterations=cast(int, 1.5),
        )
