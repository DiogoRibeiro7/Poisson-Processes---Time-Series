"""Approximate parameter estimation for Poisson log-Gaussian state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]

from .poisson_log_gaussian import (
    PoissonLaplaceFilterResult,
    _validate_integer,
    _validate_phi,
    _validate_positive,
    _validate_real,
    filter_poisson_log_gaussian,
)
from .poisson_regression import _as_counts

FloatArray = NDArray[np.float64]


def _transform_parameters(theta: FloatArray) -> tuple[float, float, float]:
    """Map unconstrained parameters to the stationary model space."""
    state_mean = float(theta[0])
    phi = float(np.tanh(theta[1]))
    state_sd = float(np.exp(np.clip(theta[2], -20.0, 5.0)))
    return state_mean, phi, state_sd


def _inverse_transform(
    state_mean: float,
    phi: float,
    state_sd: float,
) -> FloatArray:
    """Map valid model parameters to the unconstrained optimization scale."""
    return np.asarray(
        [
            state_mean,
            np.arctanh(phi),
            np.log(state_sd),
        ],
        dtype=np.float64,
    )


def _laplace_increment(
    count: int,
    *,
    prior_mean: float,
    prior_variance: float,
    posterior_mode: float,
) -> float:
    """Approximate one predictive log density using Laplace integration."""
    intensity = float(np.exp(np.clip(posterior_mode, -700.0, 700.0)))
    curvature = intensity + 1.0 / prior_variance
    centered = posterior_mode - prior_mean

    return float(
        float(count) * posterior_mode
        - intensity
        - lgamma(float(count) + 1.0)
        - 0.5 * np.log(prior_variance)
        - 0.5 * centered**2 / prior_variance
        - 0.5 * np.log(curvature)
    )


def laplace_log_marginal_likelihood(
    counts: ArrayLike,
    *,
    state_mean: float,
    phi: float,
    state_sd: float,
    filter_tolerance: float = 1e-10,
    filter_max_iterations: int = 100,
) -> float:
    """Return the sequential Laplace approximation to the marginal log likelihood."""
    result = filter_poisson_log_gaussian(
        counts,
        state_mean=state_mean,
        phi=phi,
        state_sd=state_sd,
        tolerance=filter_tolerance,
        max_iterations=filter_max_iterations,
    )
    if not result.converged:
        raise RuntimeError("Laplace filtering did not converge for all observations.")

    total = 0.0
    for index, count in enumerate(result.observed_counts):
        total += _laplace_increment(
            int(count),
            prior_mean=float(result.predicted_state_mean[index]),
            prior_variance=float(result.predicted_state_variance[index]),
            posterior_mode=float(result.filtered_state_mode[index]),
        )

    return float(total)


@dataclass(frozen=True, slots=True)
class PoissonLogGaussianParameterFit:
    """Approximate maximum-marginal-likelihood fit of state-space parameters."""

    state_mean: float
    phi: float
    state_sd: float
    approximate_log_likelihood: float
    approximate_aic: float
    approximate_bic: float
    converged: bool
    iterations: int
    optimizer_message: str
    filter_result: PoissonLaplaceFilterResult

    def __post_init__(self) -> None:
        """Validate the fitted parameter triple."""
        _validate_real(self.state_mean, name="state_mean")
        _validate_phi(self.phi)
        _validate_positive(self.state_sd, name="state_sd")
        if not np.isfinite(self.approximate_log_likelihood):
            raise ValueError("approximate_log_likelihood must be finite.")
        if not np.isfinite(self.approximate_aic):
            raise ValueError("approximate_aic must be finite.")
        if not np.isfinite(self.approximate_bic):
            raise ValueError("approximate_bic must be finite.")

    @property
    def nobs(self) -> int:
        """Return the number of fitted observations."""
        return self.filter_result.nobs

    @property
    def nparams(self) -> int:
        """Return the number of estimated state-space parameters."""
        return 3


def fit_poisson_log_gaussian_parameters(
    counts: ArrayLike,
    *,
    initial_state_mean: float | None = None,
    initial_phi: float = 0.5,
    initial_state_sd: float = 0.3,
    optimizer_tolerance: float = 1e-8,
    optimizer_max_iterations: int = 300,
    filter_tolerance: float = 1e-10,
    filter_max_iterations: int = 100,
) -> PoissonLogGaussianParameterFit:
    """Estimate state-space parameters by approximate marginal likelihood."""
    y = _as_counts(counts)
    if y.size < 5:
        raise ValueError("at least five counts are required for parameter estimation.")

    optimizer_tol = _validate_positive(
        optimizer_tolerance,
        name="optimizer_tolerance",
    )
    optimizer_iterations = _validate_integer(
        optimizer_max_iterations,
        name="optimizer_max_iterations",
        minimum=1,
    )
    filter_tol = _validate_positive(filter_tolerance, name="filter_tolerance")
    filter_iterations = _validate_integer(
        filter_max_iterations,
        name="filter_max_iterations",
        minimum=1,
    )

    phi_start = _validate_phi(initial_phi)
    sd_start = _validate_positive(initial_state_sd, name="initial_state_sd")

    if initial_state_mean is None:
        mean_start = float(np.log(max(float(np.mean(y)), 0.1)))
    else:
        mean_start = _validate_real(initial_state_mean, name="initial_state_mean")

    theta_start = _inverse_transform(mean_start, phi_start, sd_start)

    def objective(theta: FloatArray) -> float:
        state_mean, phi, state_sd = _transform_parameters(theta)
        try:
            value = laplace_log_marginal_likelihood(
                y,
                state_mean=state_mean,
                phi=phi,
                state_sd=state_sd,
                filter_tolerance=filter_tol,
                filter_max_iterations=filter_iterations,
            )
        except (FloatingPointError, RuntimeError, ValueError):
            return float("inf")

        if not np.isfinite(value):
            return float("inf")
        return -value

    optimization: Any = minimize(
        objective,
        theta_start,
        method="L-BFGS-B",
        options={
            "maxiter": optimizer_iterations,
            "ftol": optimizer_tol,
        },
    )

    theta_hat = np.asarray(optimization.x, dtype=np.float64)
    state_mean_hat, phi_hat, state_sd_hat = _transform_parameters(theta_hat)
    filter_result = filter_poisson_log_gaussian(
        y,
        state_mean=state_mean_hat,
        phi=phi_hat,
        state_sd=state_sd_hat,
        tolerance=filter_tol,
        max_iterations=filter_iterations,
    )
    approximate_log_likelihood = laplace_log_marginal_likelihood(
        y,
        state_mean=state_mean_hat,
        phi=phi_hat,
        state_sd=state_sd_hat,
        filter_tolerance=filter_tol,
        filter_max_iterations=filter_iterations,
    )

    parameter_count = 3
    approximate_aic = (
        2.0 * parameter_count - 2.0 * approximate_log_likelihood
    )
    approximate_bic = (
        np.log(y.size) * parameter_count - 2.0 * approximate_log_likelihood
    )

    return PoissonLogGaussianParameterFit(
        state_mean=state_mean_hat,
        phi=phi_hat,
        state_sd=state_sd_hat,
        approximate_log_likelihood=approximate_log_likelihood,
        approximate_aic=float(approximate_aic),
        approximate_bic=float(approximate_bic),
        converged=bool(optimization.success) and filter_result.converged,
        iterations=int(optimization.nit),
        optimizer_message=str(optimization.message),
        filter_result=filter_result,
    )
