"""Poisson log-Gaussian state-space models with Laplace filtering."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

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


def _validate_real(value: float, *, name: str) -> float:
    """Validate a finite real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    validated = float(value)
    if not np.isfinite(validated):
        raise ValueError(f"{name} must be finite.")
    return validated


def _validate_positive(value: float, *, name: str) -> float:
    """Validate a finite strictly positive value."""
    validated = _validate_real(value, name=name)
    if validated <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return validated


def _validate_phi(phi: float) -> float:
    """Validate a stationary AR(1) coefficient."""
    validated = _validate_real(phi, name="phi")
    if abs(validated) >= 1.0:
        raise ValueError("phi must satisfy abs(phi) < 1.")
    return validated


def _validate_integer(value: int, *, name: str, minimum: int) -> int:
    """Validate an integer control."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True, slots=True)
class PoissonLogGaussianSimulation:
    """Exact simulation from a Poisson log-Gaussian AR(1) state-space model."""

    counts: IntArray
    latent_state: FloatArray
    intensity: FloatArray
    state_mean: float
    phi: float
    state_sd: float

    def __post_init__(self) -> None:
        """Validate and freeze simulation arrays."""
        if self.counts.ndim != 1 or self.latent_state.ndim != 1 or self.intensity.ndim != 1:
            raise ValueError("simulation arrays must be one-dimensional.")
        if self.counts.size == 0:
            raise ValueError("simulation arrays must not be empty.")
        if self.counts.size != self.latent_state.size or self.counts.size != self.intensity.size:
            raise ValueError("simulation arrays must have equal length.")
        if np.any(self.counts < 0):
            raise ValueError("counts must be non-negative.")
        if not np.all(np.isfinite(self.latent_state)):
            raise ValueError("latent_state must be finite.")
        if np.any(self.intensity <= 0.0) or not np.all(np.isfinite(self.intensity)):
            raise ValueError("intensity must contain finite positive values.")

        _validate_real(self.state_mean, name="state_mean")
        _validate_phi(self.phi)
        _validate_positive(self.state_sd, name="state_sd")

        object.__setattr__(self, "counts", _freeze_int(self.counts))
        object.__setattr__(self, "latent_state", _freeze_float(self.latent_state))
        object.__setattr__(self, "intensity", _freeze_float(self.intensity))

    @property
    def nobs(self) -> int:
        """Return the number of retained observations."""
        return int(self.counts.size)


@dataclass(frozen=True, slots=True)
class PoissonLaplaceFilterResult:
    """Approximate filtering result for a Poisson log-Gaussian model."""

    observed_counts: IntArray
    predicted_state_mean: FloatArray
    predicted_state_variance: FloatArray
    filtered_state_mode: FloatArray
    filtered_state_variance: FloatArray
    filtered_intensity: FloatArray
    state_mean: float
    phi: float
    state_sd: float
    newton_iterations: IntArray
    converged: bool

    def __post_init__(self) -> None:
        """Validate and freeze filter arrays."""
        sizes = {
            self.observed_counts.size,
            self.predicted_state_mean.size,
            self.predicted_state_variance.size,
            self.filtered_state_mode.size,
            self.filtered_state_variance.size,
            self.filtered_intensity.size,
            self.newton_iterations.size,
        }
        if len(sizes) != 1 or next(iter(sizes)) == 0:
            raise ValueError("filter arrays must have the same non-zero length.")
        if np.any(self.predicted_state_variance <= 0.0):
            raise ValueError("predicted_state_variance must be positive.")
        if np.any(self.filtered_state_variance <= 0.0):
            raise ValueError("filtered_state_variance must be positive.")
        if np.any(self.filtered_intensity <= 0.0):
            raise ValueError("filtered_intensity must be positive.")

        _validate_real(self.state_mean, name="state_mean")
        _validate_phi(self.phi)
        _validate_positive(self.state_sd, name="state_sd")

        object.__setattr__(self, "observed_counts", _freeze_int(self.observed_counts))
        object.__setattr__(self, "predicted_state_mean", _freeze_float(self.predicted_state_mean))
        object.__setattr__(
            self,
            "predicted_state_variance",
            _freeze_float(self.predicted_state_variance),
        )
        object.__setattr__(self, "filtered_state_mode", _freeze_float(self.filtered_state_mode))
        object.__setattr__(
            self,
            "filtered_state_variance",
            _freeze_float(self.filtered_state_variance),
        )
        object.__setattr__(self, "filtered_intensity", _freeze_float(self.filtered_intensity))
        object.__setattr__(self, "newton_iterations", _freeze_int(self.newton_iterations))

    @property
    def nobs(self) -> int:
        """Return the number of filtered observations."""
        return int(self.observed_counts.size)

    def forecast_state(self, steps: int) -> tuple[FloatArray, FloatArray]:
        """Forecast latent Gaussian state mean and variance."""
        horizon = _validate_integer(steps, name="steps", minimum=1)
        means = np.empty(horizon, dtype=np.float64)
        variances = np.empty(horizon, dtype=np.float64)

        previous_mean = float(self.filtered_state_mode[-1])
        previous_variance = float(self.filtered_state_variance[-1])
        innovation_variance = self.state_sd**2

        for index in range(horizon):
            next_mean = self.state_mean + self.phi * (previous_mean - self.state_mean)
            next_variance = self.phi**2 * previous_variance + innovation_variance
            means[index] = next_mean
            variances[index] = next_variance
            previous_mean = next_mean
            previous_variance = next_variance

        return _freeze_float(means), _freeze_float(variances)

    def forecast_mean(self, steps: int) -> FloatArray:
        """Forecast count means under the Gaussian latent-state approximation."""
        state_mean, state_variance = self.forecast_state(steps)
        count_mean = np.exp(np.clip(state_mean + 0.5 * state_variance, -700.0, 700.0))
        return _freeze_float(count_mean)


def simulate_poisson_log_gaussian(
    nobs: int,
    *,
    state_mean: float,
    phi: float,
    state_sd: float,
    burn_in: int = 200,
    initial_state: float | None = None,
    rng: np.random.Generator | None = None,
) -> PoissonLogGaussianSimulation:
    """Simulate a Poisson log-Gaussian AR(1) state-space process."""
    observations = _validate_integer(nobs, name="nobs", minimum=1)
    burn = _validate_integer(burn_in, name="burn_in", minimum=0)
    mean_value = _validate_real(state_mean, name="state_mean")
    phi_value = _validate_phi(phi)
    sd_value = _validate_positive(state_sd, name="state_sd")

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")
    generator = np.random.default_rng() if rng is None else rng

    if initial_state is None:
        stationary_sd = sd_value / np.sqrt(1.0 - phi_value**2)
        current_state = float(generator.normal(mean_value, stationary_sd))
    else:
        current_state = _validate_real(initial_state, name="initial_state")

    total = observations + burn
    latent = np.empty(total, dtype=np.float64)
    intensity = np.empty(total, dtype=np.float64)
    counts = np.empty(total, dtype=np.int64)

    for index in range(total):
        if index > 0:
            current_state = (
                mean_value
                + phi_value * (current_state - mean_value)
                + float(generator.normal(0.0, sd_value))
            )
        latent[index] = current_state
        intensity[index] = float(np.exp(np.clip(current_state, -700.0, 700.0)))
        counts[index] = generator.poisson(intensity[index])

    return PoissonLogGaussianSimulation(
        counts=counts[burn:],
        latent_state=latent[burn:],
        intensity=intensity[burn:],
        state_mean=mean_value,
        phi=phi_value,
        state_sd=sd_value,
    )


def _laplace_update(
    count: int,
    *,
    prior_mean: float,
    prior_variance: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, float, int, bool]:
    """Compute one Laplace posterior mode/variance update by Newton iteration."""
    mode = prior_mean

    for iteration in range(1, max_iterations + 1):
        intensity = float(np.exp(np.clip(mode, -700.0, 700.0)))
        gradient = float(count) - intensity - (mode - prior_mean) / prior_variance
        hessian = -intensity - 1.0 / prior_variance
        step = gradient / hessian
        new_mode = mode - step

        if abs(new_mode - mode) <= tolerance * (1.0 + abs(new_mode)):
            curvature = (
                float(np.exp(np.clip(new_mode, -700.0, 700.0)))
                + 1.0 / prior_variance
            )
            variance = 1.0 / curvature
            return new_mode, variance, iteration, True
        mode = new_mode

    variance = 1.0 / (float(np.exp(np.clip(mode, -700.0, 700.0))) + 1.0 / prior_variance)
    return mode, variance, max_iterations, False


def filter_poisson_log_gaussian(
    counts: ArrayLike,
    *,
    state_mean: float,
    phi: float,
    state_sd: float,
    initial_state_mean: float | None = None,
    initial_state_variance: float | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> PoissonLaplaceFilterResult:
    """Approximate latent-state filtering using a Laplace Gaussian update."""
    y = _as_counts(counts)
    mean_value = _validate_real(state_mean, name="state_mean")
    phi_value = _validate_phi(phi)
    sd_value = _validate_positive(state_sd, name="state_sd")
    tolerance_value = _validate_positive(tolerance, name="tolerance")
    iterations_limit = _validate_integer(max_iterations, name="max_iterations", minimum=1)

    stationary_variance = sd_value**2 / (1.0 - phi_value**2)
    if initial_state_mean is None:
        previous_mean = mean_value
    else:
        previous_mean = _validate_real(initial_state_mean, name="initial_state_mean")

    if initial_state_variance is None:
        previous_variance = stationary_variance
    else:
        previous_variance = _validate_positive(
            initial_state_variance,
            name="initial_state_variance",
        )

    predicted_mean = np.empty(y.size, dtype=np.float64)
    predicted_variance = np.empty(y.size, dtype=np.float64)
    filtered_mode = np.empty(y.size, dtype=np.float64)
    filtered_variance = np.empty(y.size, dtype=np.float64)
    filtered_intensity = np.empty(y.size, dtype=np.float64)
    iterations = np.empty(y.size, dtype=np.int64)

    all_converged = True
    innovation_variance = sd_value**2

    for index, count in enumerate(y):
        if index == 0:
            prior_mean = previous_mean
            prior_variance = previous_variance
        else:
            prior_mean = mean_value + phi_value * (previous_mean - mean_value)
            prior_variance = phi_value**2 * previous_variance + innovation_variance

        mode, variance, used_iterations, converged = _laplace_update(
            int(count),
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            tolerance=tolerance_value,
            max_iterations=iterations_limit,
        )

        predicted_mean[index] = prior_mean
        predicted_variance[index] = prior_variance
        filtered_mode[index] = mode
        filtered_variance[index] = variance
        filtered_intensity[index] = float(
            np.exp(np.clip(mode + 0.5 * variance, -700.0, 700.0))
        )
        iterations[index] = used_iterations
        all_converged = all_converged and converged

        previous_mean = mode
        previous_variance = variance

    return PoissonLaplaceFilterResult(
        observed_counts=y,
        predicted_state_mean=predicted_mean,
        predicted_state_variance=predicted_variance,
        filtered_state_mode=filtered_mode,
        filtered_state_variance=filtered_variance,
        filtered_intensity=filtered_intensity,
        state_mean=mean_value,
        phi=phi_value,
        state_sd=sd_value,
        newton_iterations=iterations,
        converged=all_converged,
    )
