"""Backward smoothing for Laplace-Gaussian Poisson state-space models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .poisson_log_gaussian import (
    FloatArray,
    PoissonLaplaceFilterResult,
    _freeze_float,
)


@dataclass(frozen=True, slots=True)
class PoissonLaplaceSmootherResult:
    """Approximate RTS-style smoother built on Laplace filter moments."""

    smoothed_state_mean: FloatArray
    smoothed_state_variance: FloatArray
    smoothed_intensity: FloatArray
    smoothing_gain: FloatArray
    filter_result: PoissonLaplaceFilterResult

    def __post_init__(self) -> None:
        """Validate and freeze smoother arrays."""
        sizes = {
            self.smoothed_state_mean.size,
            self.smoothed_state_variance.size,
            self.smoothed_intensity.size,
            self.smoothing_gain.size,
            self.filter_result.nobs,
        }
        if len(sizes) != 1 or next(iter(sizes)) == 0:
            raise ValueError("smoother arrays must match the filter length.")
        if np.any(self.smoothed_state_variance <= 0.0):
            raise ValueError("smoothed_state_variance must be positive.")
        if np.any(self.smoothed_intensity <= 0.0):
            raise ValueError("smoothed_intensity must be positive.")
        if not np.all(np.isfinite(self.smoothed_state_mean)):
            raise ValueError("smoothed_state_mean must be finite.")
        if not np.all(np.isfinite(self.smoothed_state_variance)):
            raise ValueError("smoothed_state_variance must be finite.")
        if not np.all(np.isfinite(self.smoothed_intensity)):
            raise ValueError("smoothed_intensity must be finite.")
        if not np.all(np.isfinite(self.smoothing_gain)):
            raise ValueError("smoothing_gain must be finite.")

        object.__setattr__(
            self,
            "smoothed_state_mean",
            _freeze_float(self.smoothed_state_mean),
        )
        object.__setattr__(
            self,
            "smoothed_state_variance",
            _freeze_float(self.smoothed_state_variance),
        )
        object.__setattr__(
            self,
            "smoothed_intensity",
            _freeze_float(self.smoothed_intensity),
        )
        object.__setattr__(
            self,
            "smoothing_gain",
            _freeze_float(self.smoothing_gain),
        )

    @property
    def nobs(self) -> int:
        """Return the number of smoothed observations."""
        return int(self.smoothed_state_mean.size)


def smooth_poisson_log_gaussian(
    filter_result: PoissonLaplaceFilterResult,
) -> PoissonLaplaceSmootherResult:
    """Apply an RTS-style backward smoother to Laplace filter moments.

    The forward Poisson update is approximate, but the backward AR(1) recursion
    is analytic conditional on those Gaussian approximations.
    """
    if not isinstance(filter_result, PoissonLaplaceFilterResult):
        raise TypeError("filter_result must be a PoissonLaplaceFilterResult.")
    if not filter_result.converged:
        raise ValueError("filter_result must have converged before smoothing.")

    nobs = filter_result.nobs
    means = np.array(filter_result.filtered_state_mode, dtype=np.float64, copy=True)
    variances = np.array(
        filter_result.filtered_state_variance,
        dtype=np.float64,
        copy=True,
    )
    gains = np.zeros(nobs, dtype=np.float64)

    for index in range(nobs - 2, -1, -1):
        predicted_variance = float(
            filter_result.predicted_state_variance[index + 1]
        )
        if predicted_variance <= 0.0:
            raise ValueError("predicted state variance must be positive.")

        gain = (
            float(filter_result.filtered_state_variance[index])
            * filter_result.phi
            / predicted_variance
        )
        gains[index] = gain

        prediction_mean = float(filter_result.predicted_state_mean[index + 1])
        means[index] = (
            float(filter_result.filtered_state_mode[index])
            + gain * (means[index + 1] - prediction_mean)
        )
        variances[index] = (
            float(filter_result.filtered_state_variance[index])
            + gain**2
            * (
                variances[index + 1]
                - predicted_variance
            )
        )
        if variances[index] <= 0.0:
            raise FloatingPointError(
                "non-positive smoothed state variance encountered."
            )

    intensity = np.exp(
        np.clip(means + 0.5 * variances, -700.0, 700.0)
    )

    return PoissonLaplaceSmootherResult(
        smoothed_state_mean=means,
        smoothed_state_variance=variances,
        smoothed_intensity=np.asarray(intensity, dtype=np.float64),
        smoothing_gain=gains,
        filter_result=filter_result,
    )
