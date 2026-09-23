"""Poisson INGARCH models for serially dependent count time series."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]

from .poisson_regression import IntArray, _as_counts

FloatArray = NDArray[np.float64]


def _freeze_float(values: ArrayLike) -> FloatArray:
    """Return an immutable float64 copy."""
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _freeze_int(values: ArrayLike) -> IntArray:
    """Return an immutable int64 copy."""
    array = np.array(values, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _validate_positive_real(value: Real, *, name: str) -> float:
    """Validate a strictly positive finite real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real number.")

    validated = float(value)
    if not np.isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return validated


def _validate_coefficient(value: Real, *, name: str) -> float:
    """Validate a non-negative finite INGARCH coefficient."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative real number.")

    validated = float(value)
    if not np.isfinite(validated) or validated < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return validated


def _validate_dynamics(omega: Real, alpha: Real, beta: Real) -> tuple[float, float, float]:
    """Validate stable INGARCH(1,1) parameters."""
    omega_value = _validate_positive_real(omega, name="omega")
    alpha_value = _validate_coefficient(alpha, name="alpha")
    beta_value = _validate_coefficient(beta, name="beta")

    if alpha_value + beta_value >= 1.0:
        raise ValueError("alpha + beta must be strictly smaller than one.")

    return omega_value, alpha_value, beta_value


def _validate_integer(value: int, *, name: str, minimum: int) -> int:
    """Validate an integer control parameter."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _transform_parameters(theta: FloatArray) -> tuple[float, float, float]:
    """Map unconstrained optimisation parameters to the stationary region."""
    omega = float(np.exp(np.clip(theta[0], -700.0, 700.0)))

    maximum = max(0.0, float(theta[1]), float(theta[2]))
    baseline = np.exp(-maximum)
    count_weight = np.exp(float(theta[1]) - maximum)
    intensity_weight = np.exp(float(theta[2]) - maximum)
    denominator = baseline + count_weight + intensity_weight

    alpha = float(count_weight / denominator)
    beta = float(intensity_weight / denominator)
    return omega, alpha, beta


def _inverse_transform(omega: float, alpha: float, beta: float) -> FloatArray:
    """Map valid INGARCH parameters to the unconstrained optimisation scale."""
    remainder = 1.0 - alpha - beta
    return np.asarray(
        [
            np.log(omega),
            np.log(alpha / remainder),
            np.log(beta / remainder),
        ],
        dtype=np.float64,
    )


def _conditional_intensity(
    counts: IntArray,
    *,
    omega: float,
    alpha: float,
    beta: float,
    initial_intensity: float,
) -> FloatArray:
    """Compute the conditional-intensity recursion for observed counts."""
    intensity = np.empty(counts.size, dtype=np.float64)
    intensity[0] = initial_intensity

    for index in range(1, counts.size):
        intensity[index] = (
            omega
            + alpha * float(counts[index - 1])
            + beta * intensity[index - 1]
        )

    return intensity


def _conditional_log_likelihood(counts: IntArray, intensity: FloatArray) -> float:
    """Compute conditional Poisson log-likelihood excluding the first observation."""
    total = 0.0
    for count, expected in zip(counts[1:], intensity[1:], strict=True):
        y = float(count)
        total += y * np.log(expected) - expected - lgamma(y + 1.0)
    return float(total)


def _negative_log_likelihood(
    theta: FloatArray,
    counts: IntArray,
    initial_intensity: float,
) -> float:
    """Objective on the unconstrained parameter scale."""
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

    return -_conditional_log_likelihood(counts, intensity)


@dataclass(frozen=True, slots=True)
class PoissonINGARCHSimulation:
    """Simulated Poisson INGARCH(1,1) path."""

    counts: IntArray
    intensity: FloatArray
    omega: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        """Freeze simulation arrays."""
        if self.counts.ndim != 1 or self.intensity.ndim != 1:
            raise ValueError("counts and intensity must be one-dimensional.")
        if self.counts.size == 0 or self.counts.size != self.intensity.size:
            raise ValueError("counts and intensity must have the same non-zero length.")
        if np.any(self.counts < 0):
            raise ValueError("counts must be non-negative.")
        if np.any(self.intensity <= 0.0) or not np.all(np.isfinite(self.intensity)):
            raise ValueError("intensity must contain finite positive values.")

        _validate_dynamics(self.omega, self.alpha, self.beta)
        object.__setattr__(self, "counts", _freeze_int(self.counts))
        object.__setattr__(self, "intensity", _freeze_float(self.intensity))

    @property
    def nobs(self) -> int:
        """Return the number of retained observations."""
        return int(self.counts.size)


@dataclass(frozen=True, slots=True)
class PoissonINGARCHResult:
    """Conditional maximum-likelihood fit for Poisson INGARCH(1,1)."""

    omega: float
    alpha: float
    beta: float
    observed_counts: IntArray
    fitted_intensity: FloatArray
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
        _validate_positive_real(self.initial_intensity, name="initial_intensity")

        object.__setattr__(self, "observed_counts", _freeze_int(self.observed_counts))
        object.__setattr__(self, "fitted_intensity", _freeze_float(self.fitted_intensity))

    @property
    def nobs(self) -> int:
        """Return the number of observed counts."""
        return int(self.observed_counts.size)

    @property
    def unconditional_mean(self) -> float:
        """Return the stationary unconditional mean implied by the fit."""
        return self.omega / (1.0 - self.alpha - self.beta)

    def forecast_mean(self, steps: int) -> FloatArray:
        """Forecast conditional means using the INGARCH recursion."""
        horizon = _validate_integer(steps, name="steps", minimum=1)
        forecasts = np.empty(horizon, dtype=np.float64)

        first = (
            self.omega
            + self.alpha * float(self.observed_counts[-1])
            + self.beta * float(self.fitted_intensity[-1])
        )
        forecasts[0] = first

        persistence = self.alpha + self.beta
        for index in range(1, horizon):
            forecasts[index] = self.omega + persistence * forecasts[index - 1]

        return _freeze_float(forecasts)


def simulate_poisson_ingarch(
    nobs: int,
    *,
    omega: Real,
    alpha: Real,
    beta: Real,
    burn_in: int = 200,
    initial_intensity: Real | None = None,
    rng: np.random.Generator | None = None,
) -> PoissonINGARCHSimulation:
    """Simulate a stationary Poisson INGARCH(1,1) process."""
    observations = _validate_integer(nobs, name="nobs", minimum=1)
    burn = _validate_integer(burn_in, name="burn_in", minimum=0)
    omega_value, alpha_value, beta_value = _validate_dynamics(omega, alpha, beta)

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

    intensity[0] = initial
    counts[0] = generator.poisson(intensity[0])

    for index in range(1, total):
        intensity[index] = (
            omega_value
            + alpha_value * float(counts[index - 1])
            + beta_value * intensity[index - 1]
        )
        counts[index] = generator.poisson(intensity[index])

    return PoissonINGARCHSimulation(
        counts=counts[burn:],
        intensity=intensity[burn:],
        omega=omega_value,
        alpha=alpha_value,
        beta=beta_value,
    )


def fit_poisson_ingarch(
    counts: ArrayLike,
    *,
    initial_intensity: Real | None = None,
    max_iterations: int = 1000,
    tolerance: Real = 1e-8,
) -> PoissonINGARCHResult:
    """Fit Poisson INGARCH(1,1) by conditional maximum likelihood."""
    y = _as_counts(counts)
    if y.size < 3:
        raise ValueError("at least three counts are required for INGARCH fitting.")

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
        args=(y, initial),
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
    log_likelihood = _conditional_log_likelihood(y, intensity)

    effective_nobs = y.size - 1
    parameter_count = 3
    aic = 2.0 * parameter_count - 2.0 * log_likelihood
    bic = np.log(effective_nobs) * parameter_count - 2.0 * log_likelihood

    return PoissonINGARCHResult(
        omega=omega_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        observed_counts=y,
        fitted_intensity=intensity,
        initial_intensity=initial,
        log_likelihood=log_likelihood,
        aic=float(aic),
        bic=float(bic),
        converged=bool(optimization.success),
        iterations=int(optimization.nit),
        optimizer_message=str(optimization.message),
    )
