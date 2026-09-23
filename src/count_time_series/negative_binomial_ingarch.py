"""Negative-binomial INGARCH models for overdispersed count time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize  # type: ignore[import-untyped]

from .ingarch import (
    FloatArray,
    _conditional_intensity,
    _freeze_float,
    _freeze_int,
    _inverse_transform,
    _transform_parameters,
    _validate_dynamics,
    _validate_integer,
    _validate_positive_real,
)
from .negative_binomial import _negative_binomial_log_likelihood, _validate_dispersion
from .poisson_regression import IntArray, _as_counts


def _negative_log_likelihood(
    theta: FloatArray,
    counts: IntArray,
    initial_intensity: float,
    dispersion: float,
) -> float:
    """Conditional NB2 objective on the unconstrained dynamics scale."""
    omega, alpha, beta = _transform_parameters(theta)
    intensity = _conditional_intensity(
        counts,
        omega=omega,
        alpha=alpha,
        beta=beta,
        initial_intensity=initial_intensity,
    )

    if not np.all(np.isfinite(intensity)) or np.any(intensity <= 0.0):
        return float("inf")

    return -_negative_binomial_log_likelihood(
        counts[1:],
        intensity[1:],
        dispersion=dispersion,
    )


@dataclass(frozen=True, slots=True)
class NegativeBinomialINGARCHSimulation:
    """Simulated fixed-dispersion NB2 INGARCH(1,1) path."""

    counts: IntArray
    intensity: FloatArray
    conditional_variance: FloatArray
    omega: float
    alpha: float
    beta: float
    dispersion: float

    def __post_init__(self) -> None:
        """Validate parameters and freeze simulation arrays."""
        if self.counts.ndim != 1:
            raise ValueError("counts must be one-dimensional.")
        if self.intensity.ndim != 1 or self.conditional_variance.ndim != 1:
            raise ValueError("intensity and conditional_variance must be one-dimensional.")
        if self.counts.size == 0:
            raise ValueError("simulation arrays must not be empty.")
        if self.counts.size != self.intensity.size:
            raise ValueError("counts and intensity must have equal length.")
        if self.counts.size != self.conditional_variance.size:
            raise ValueError("counts and conditional_variance must have equal length.")
        if np.any(self.counts < 0):
            raise ValueError("counts must be non-negative.")
        if np.any(self.intensity <= 0.0) or not np.all(np.isfinite(self.intensity)):
            raise ValueError("intensity must contain finite positive values.")
        if np.any(self.conditional_variance <= 0.0) or not np.all(
            np.isfinite(self.conditional_variance)
        ):
            raise ValueError("conditional_variance must contain finite positive values.")

        _validate_dynamics(self.omega, self.alpha, self.beta)
        _validate_dispersion(self.dispersion)

        object.__setattr__(self, "counts", _freeze_int(self.counts))
        object.__setattr__(self, "intensity", _freeze_float(self.intensity))
        object.__setattr__(
            self,
            "conditional_variance",
            _freeze_float(self.conditional_variance),
        )

    @property
    def nobs(self) -> int:
        """Return the number of retained observations."""
        return int(self.counts.size)


@dataclass(frozen=True, slots=True)
class NegativeBinomialINGARCHResult:
    """Conditional MLE for NB2 INGARCH(1,1) with fixed dispersion."""

    omega: float
    alpha: float
    beta: float
    dispersion: float
    observed_counts: IntArray
    fitted_intensity: FloatArray
    fitted_conditional_variance: FloatArray
    initial_intensity: float
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    iterations: int
    optimizer_message: str

    def __post_init__(self) -> None:
        """Validate parameters and freeze fitted arrays."""
        _validate_dynamics(self.omega, self.alpha, self.beta)
        _validate_dispersion(self.dispersion)
        _validate_positive_real(self.initial_intensity, name="initial_intensity")

        object.__setattr__(self, "observed_counts", _freeze_int(self.observed_counts))
        object.__setattr__(self, "fitted_intensity", _freeze_float(self.fitted_intensity))
        object.__setattr__(
            self,
            "fitted_conditional_variance",
            _freeze_float(self.fitted_conditional_variance),
        )

    @property
    def nobs(self) -> int:
        """Return the number of observed counts."""
        return int(self.observed_counts.size)

    @property
    def unconditional_mean(self) -> float:
        """Return the stationary unconditional mean implied by the dynamics."""
        return self.omega / (1.0 - self.alpha - self.beta)

    def forecast_mean(self, steps: int) -> FloatArray:
        """Forecast conditional means using the INGARCH recursion."""
        horizon = _validate_integer(steps, name="steps", minimum=1)
        forecasts = np.empty(horizon, dtype=np.float64)

        forecasts[0] = (
            self.omega
            + self.alpha * float(self.observed_counts[-1])
            + self.beta * float(self.fitted_intensity[-1])
        )

        persistence = self.alpha + self.beta
        for index in range(1, horizon):
            forecasts[index] = self.omega + persistence * forecasts[index - 1]

        return _freeze_float(forecasts)

    def one_step_forecast_variance(self) -> float:
        """Return the NB2 variance for the one-step-ahead conditional forecast."""
        mean = float(self.forecast_mean(1)[0])
        return mean + self.dispersion * mean**2


def simulate_negative_binomial_ingarch(
    nobs: int,
    *,
    omega: float,
    alpha: float,
    beta: float,
    dispersion: float,
    burn_in: int = 200,
    initial_intensity: float | None = None,
    rng: np.random.Generator | None = None,
) -> NegativeBinomialINGARCHSimulation:
    """Simulate a stationary fixed-dispersion NB2 INGARCH(1,1) process."""
    observations = _validate_integer(nobs, name="nobs", minimum=1)
    burn = _validate_integer(burn_in, name="burn_in", minimum=0)
    omega_value, alpha_value, beta_value = _validate_dynamics(omega, alpha, beta)
    dispersion_value = _validate_dispersion(dispersion)

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")

    if initial_intensity is None:
        initial = omega_value / (1.0 - alpha_value - beta_value)
    else:
        initial = _validate_positive_real(
            initial_intensity,
            name="initial_intensity",
        )

    generator = np.random.default_rng() if rng is None else rng
    total = observations + burn
    counts = np.empty(total, dtype=np.int64)
    intensity = np.empty(total, dtype=np.float64)
    conditional_variance = np.empty(total, dtype=np.float64)
    size = 1.0 / dispersion_value

    intensity[0] = initial
    conditional_variance[0] = initial + dispersion_value * initial**2
    probability = size / (size + initial)
    counts[0] = generator.negative_binomial(size, probability)

    for index in range(1, total):
        intensity[index] = (
            omega_value
            + alpha_value * float(counts[index - 1])
            + beta_value * intensity[index - 1]
        )
        conditional_variance[index] = (
            intensity[index] + dispersion_value * intensity[index] ** 2
        )
        probability = size / (size + intensity[index])
        counts[index] = generator.negative_binomial(size, probability)

    return NegativeBinomialINGARCHSimulation(
        counts=counts[burn:],
        intensity=intensity[burn:],
        conditional_variance=conditional_variance[burn:],
        omega=omega_value,
        alpha=alpha_value,
        beta=beta_value,
        dispersion=dispersion_value,
    )


def fit_negative_binomial_ingarch(
    counts: ArrayLike,
    *,
    dispersion: float,
    initial_intensity: float | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> NegativeBinomialINGARCHResult:
    """Fit NB2 INGARCH(1,1) by conditional MLE for fixed dispersion."""
    y = _as_counts(counts)
    if y.size < 3:
        raise ValueError("at least three counts are required for INGARCH fitting.")

    dispersion_value = _validate_dispersion(dispersion)
    iterations_limit = _validate_integer(
        max_iterations,
        name="max_iterations",
        minimum=1,
    )
    tolerance_value = _validate_positive_real(tolerance, name="tolerance")

    if initial_intensity is None:
        initial = max(float(np.mean(y)), 0.1)
    else:
        initial = _validate_positive_real(
            initial_intensity,
            name="initial_intensity",
        )

    sample_mean = max(float(np.mean(y)), 0.1)
    alpha_start = 0.10
    beta_start = 0.50
    omega_start = max(sample_mean * (1.0 - alpha_start - beta_start), 0.1)
    theta_start = _inverse_transform(omega_start, alpha_start, beta_start)

    optimization: Any = minimize(
        _negative_log_likelihood,
        theta_start,
        args=(y, initial, dispersion_value),
        method="L-BFGS-B",
        options={
            "maxiter": iterations_limit,
            "ftol": tolerance_value,
        },
    )

    theta_hat = np.asarray(optimization.x, dtype=np.float64)
    omega_hat, alpha_hat, beta_hat = _transform_parameters(theta_hat)
    intensity = _conditional_intensity(
        y,
        omega=omega_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        initial_intensity=initial,
    )
    variance = intensity + dispersion_value * intensity**2
    log_likelihood = _negative_binomial_log_likelihood(
        y[1:],
        intensity[1:],
        dispersion=dispersion_value,
    )

    effective_nobs = y.size - 1
    parameter_count = 3
    aic = 2.0 * parameter_count - 2.0 * log_likelihood
    bic = np.log(effective_nobs) * parameter_count - 2.0 * log_likelihood

    return NegativeBinomialINGARCHResult(
        omega=omega_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        dispersion=dispersion_value,
        observed_counts=y,
        fitted_intensity=intensity,
        fitted_conditional_variance=variance,
        initial_intensity=initial,
        log_likelihood=log_likelihood,
        aic=float(aic),
        bic=float(bic),
        converged=bool(optimization.success),
        iterations=int(optimization.nit),
        optimizer_message=str(optimization.message),
    )
