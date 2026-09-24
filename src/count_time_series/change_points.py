"""Single change-point estimation for piecewise-constant Poisson intensity."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

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


def _validate_min_segment_size(value: int, *, nobs: int) -> int:
    """Validate the minimum admissible segment size."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("min_segment_size must be an integer.")
    if value < 1:
        raise ValueError("min_segment_size must be at least one.")
    if 2 * value > nobs:
        raise ValueError(
            "min_segment_size leaves no admissible change-point location."
        )
    return value


def _segment_log_likelihood(
    total_count: int,
    length: int,
    log_factorial_sum: float,
) -> tuple[float, float]:
    """Return Poisson MLE intensity and maximized segment log likelihood."""
    if length < 1:
        raise ValueError("segment length must be positive.")

    rate = float(total_count) / float(length)
    if total_count == 0:
        return 0.0, -log_factorial_sum

    log_likelihood = (
        float(total_count) * np.log(rate)
        - float(length) * rate
        - log_factorial_sum
    )
    return rate, float(log_likelihood)


@dataclass(frozen=True, slots=True)
class PoissonChangePointResult:
    """Profile-likelihood fit of a single Poisson intensity change point."""

    change_point: int
    rate_before: float
    rate_after: float
    null_rate: float
    log_likelihood_change: float
    log_likelihood_null: float
    likelihood_ratio: float
    bic_change: float
    bic_null: float
    candidate_change_points: IntArray
    profile_log_likelihood: FloatArray

    def __post_init__(self) -> None:
        """Validate and freeze profile arrays."""
        if self.change_point < 1:
            raise ValueError("change_point must be positive.")
        for value, name in [
            (self.rate_before, "rate_before"),
            (self.rate_after, "rate_after"),
            (self.null_rate, "null_rate"),
        ]:
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.candidate_change_points.ndim != 1:
            raise ValueError("candidate_change_points must be one-dimensional.")
        if self.profile_log_likelihood.ndim != 1:
            raise ValueError("profile_log_likelihood must be one-dimensional.")
        if self.candidate_change_points.size != self.profile_log_likelihood.size:
            raise ValueError("profile arrays must have equal length.")
        if self.candidate_change_points.size == 0:
            raise ValueError("profile arrays must not be empty.")

        object.__setattr__(
            self,
            "candidate_change_points",
            _freeze_int(self.candidate_change_points),
        )
        object.__setattr__(
            self,
            "profile_log_likelihood",
            _freeze_float(self.profile_log_likelihood),
        )

    @property
    def bic_improvement(self) -> float:
        """Return BIC(null) - BIC(change); positive values favor the break model."""
        return self.bic_null - self.bic_change


def fit_poisson_single_change_point(
    counts: ArrayLike,
    *,
    min_segment_size: int = 5,
) -> PoissonChangePointResult:
    """Fit one unknown change point in a piecewise-constant Poisson intensity."""
    y = _as_counts(counts)
    nobs = int(y.size)
    minimum = _validate_min_segment_size(min_segment_size, nobs=nobs)

    cumulative_counts = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(y, dtype=np.int64)]
    )
    log_factorials = np.fromiter(
        (lgamma(float(count) + 1.0) for count in y),
        dtype=np.float64,
        count=nobs,
    )
    cumulative_log_factorials = np.concatenate(
        [np.array([0.0], dtype=np.float64), np.cumsum(log_factorials)]
    )

    candidates = np.arange(minimum, nobs - minimum + 1, dtype=np.int64)
    profile = np.empty(candidates.size, dtype=np.float64)

    total_count = int(cumulative_counts[-1])
    total_factorials = float(cumulative_log_factorials[-1])

    for index, tau_value in enumerate(candidates):
        tau = int(tau_value)
        left_count = int(cumulative_counts[tau])
        right_count = total_count - left_count
        left_factorials = float(cumulative_log_factorials[tau])
        right_factorials = total_factorials - left_factorials

        _, left_ll = _segment_log_likelihood(
            left_count,
            tau,
            left_factorials,
        )
        _, right_ll = _segment_log_likelihood(
            right_count,
            nobs - tau,
            right_factorials,
        )
        profile[index] = left_ll + right_ll

    best_index = int(np.argmax(profile))
    change_point = int(candidates[best_index])

    left_count = int(cumulative_counts[change_point])
    right_count = total_count - left_count
    left_factorials = float(cumulative_log_factorials[change_point])
    right_factorials = total_factorials - left_factorials

    rate_before, left_ll = _segment_log_likelihood(
        left_count,
        change_point,
        left_factorials,
    )
    rate_after, right_ll = _segment_log_likelihood(
        right_count,
        nobs - change_point,
        right_factorials,
    )
    null_rate, null_ll = _segment_log_likelihood(
        total_count,
        nobs,
        total_factorials,
    )

    change_ll = left_ll + right_ll
    likelihood_ratio = 2.0 * (change_ll - null_ll)

    null_parameter_count = 1
    change_parameter_count = 3
    bic_null = (
        np.log(nobs) * null_parameter_count - 2.0 * null_ll
    )
    bic_change = (
        np.log(nobs) * change_parameter_count - 2.0 * change_ll
    )

    return PoissonChangePointResult(
        change_point=change_point,
        rate_before=rate_before,
        rate_after=rate_after,
        null_rate=null_rate,
        log_likelihood_change=change_ll,
        log_likelihood_null=null_ll,
        likelihood_ratio=float(likelihood_ratio),
        bic_change=float(bic_change),
        bic_null=float(bic_null),
        candidate_change_points=candidates,
        profile_log_likelihood=profile,
    )
