"""Exact multiple change-point segmentation for Poisson count series."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .change_points import _segment_log_likelihood
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


def _validate_controls(
    *,
    nobs: int,
    max_change_points: int,
    min_segment_size: int,
) -> tuple[int, int]:
    """Validate segmentation controls."""
    if isinstance(max_change_points, bool) or not isinstance(max_change_points, int):
        raise TypeError("max_change_points must be an integer.")
    if max_change_points < 1:
        raise ValueError("max_change_points must be at least one.")

    if isinstance(min_segment_size, bool) or not isinstance(min_segment_size, int):
        raise TypeError("min_segment_size must be an integer.")
    if min_segment_size < 1:
        raise ValueError("min_segment_size must be at least one.")

    max_feasible = nobs // min_segment_size - 1
    if max_feasible < 1:
        raise ValueError(
            "min_segment_size leaves no feasible multiple-change-point model."
        )
    if max_change_points > max_feasible:
        raise ValueError(
            "max_change_points is too large for the requested minimum segment size."
        )

    return max_change_points, min_segment_size


def _prefix_statistics(counts: IntArray) -> tuple[IntArray, FloatArray]:
    """Return cumulative counts and cumulative log-factorial terms."""
    nobs = int(counts.size)
    cumulative_counts = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum(counts, dtype=np.int64),
        ]
    )
    log_factorials = np.fromiter(
        (lgamma(float(count) + 1.0) for count in counts),
        dtype=np.float64,
        count=nobs,
    )
    cumulative_log_factorials = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            np.cumsum(log_factorials),
        ]
    )
    return cumulative_counts, cumulative_log_factorials


def _segment_fit(
    start: int,
    end: int,
    *,
    cumulative_counts: IntArray,
    cumulative_log_factorials: FloatArray,
) -> tuple[float, float]:
    """Return the MLE rate and maximized Poisson log likelihood for [start, end)."""
    total_count = int(cumulative_counts[end] - cumulative_counts[start])
    log_factorial_sum = float(
        cumulative_log_factorials[end] - cumulative_log_factorials[start]
    )
    return _segment_log_likelihood(
        total_count,
        end - start,
        log_factorial_sum,
    )


def _reconstruct_boundaries(
    previous: IntArray,
    *,
    segments: int,
    nobs: int,
) -> IntArray:
    """Reconstruct optimal segment boundaries from dynamic-programming pointers."""
    boundaries = np.empty(segments + 1, dtype=np.int64)
    boundaries[-1] = nobs

    end = nobs
    for segment in range(segments, 0, -1):
        start = int(previous[segment, end])
        if start < 0:
            raise RuntimeError("failed to reconstruct optimal segmentation.")
        boundaries[segment - 1] = start
        end = start

    if boundaries[0] != 0:
        raise RuntimeError("optimal segmentation does not start at zero.")
    return boundaries


@dataclass(frozen=True, slots=True)
class PoissonMultipleChangePointResult:
    """BIC-selected exact dynamic-programming Poisson segmentation."""

    change_points: IntArray
    segment_boundaries: IntArray
    segment_rates: FloatArray
    selected_change_points: int
    selected_log_likelihood: float
    selected_bic: float
    change_point_counts: IntArray
    optimal_log_likelihood: FloatArray
    bic_by_change_count: FloatArray

    def __post_init__(self) -> None:
        """Validate and freeze segmentation arrays."""
        if self.selected_change_points < 0:
            raise ValueError("selected_change_points must be non-negative.")
        if self.segment_boundaries.ndim != 1:
            raise ValueError("segment_boundaries must be one-dimensional.")
        if self.segment_rates.ndim != 1:
            raise ValueError("segment_rates must be one-dimensional.")
        if self.change_points.ndim != 1:
            raise ValueError("change_points must be one-dimensional.")
        if self.segment_boundaries.size != self.segment_rates.size + 1:
            raise ValueError("segment_boundaries must be one longer than segment_rates.")
        if self.change_points.size != self.segment_rates.size - 1:
            raise ValueError("change_points must separate adjacent segments.")
        if self.change_points.size != self.selected_change_points:
            raise ValueError("selected_change_points must match change_points length.")
        if self.change_point_counts.size != self.optimal_log_likelihood.size:
            raise ValueError("model-selection arrays must have equal length.")
        if self.change_point_counts.size != self.bic_by_change_count.size:
            raise ValueError("model-selection arrays must have equal length.")
        if np.any(self.segment_rates < 0.0):
            raise ValueError("segment_rates must be non-negative.")
        if not np.all(np.isfinite(self.segment_rates)):
            raise ValueError("segment_rates must be finite.")

        object.__setattr__(self, "change_points", _freeze_int(self.change_points))
        object.__setattr__(
            self,
            "segment_boundaries",
            _freeze_int(self.segment_boundaries),
        )
        object.__setattr__(self, "segment_rates", _freeze_float(self.segment_rates))
        object.__setattr__(
            self,
            "change_point_counts",
            _freeze_int(self.change_point_counts),
        )
        object.__setattr__(
            self,
            "optimal_log_likelihood",
            _freeze_float(self.optimal_log_likelihood),
        )
        object.__setattr__(
            self,
            "bic_by_change_count",
            _freeze_float(self.bic_by_change_count),
        )

    @property
    def n_segments(self) -> int:
        """Return the selected number of segments."""
        return int(self.segment_rates.size)

    @property
    def bic_improvement_over_null(self) -> float:
        """Return BIC(no break) - BIC(selected); positive values favor segmentation."""
        return float(self.bic_by_change_count[0] - self.selected_bic)


def fit_poisson_multiple_change_points(
    counts: ArrayLike,
    *,
    max_change_points: int = 5,
    min_segment_size: int = 5,
) -> PoissonMultipleChangePointResult:
    """Fit multiple piecewise-constant Poisson intensity changes exactly.

    For each change-point count from zero through the configured maximum,
    dynamic programming finds the globally optimal segmentation subject to
    the minimum segment size. BIC then selects among those optimal fits.
    """
    y = _as_counts(counts)
    nobs = int(y.size)
    maximum, minimum = _validate_controls(
        nobs=nobs,
        max_change_points=max_change_points,
        min_segment_size=min_segment_size,
    )
    max_segments = maximum + 1

    cumulative_counts, cumulative_log_factorials = _prefix_statistics(y)

    best = np.full((max_segments + 1, nobs + 1), -np.inf, dtype=np.float64)
    previous = np.full((max_segments + 1, nobs + 1), -1, dtype=np.int64)

    for end in range(minimum, nobs + 1):
        _, log_likelihood = _segment_fit(
            0,
            end,
            cumulative_counts=cumulative_counts,
            cumulative_log_factorials=cumulative_log_factorials,
        )
        best[1, end] = log_likelihood
        previous[1, end] = 0

    for segments in range(2, max_segments + 1):
        earliest_end = segments * minimum
        earliest_start = (segments - 1) * minimum

        for end in range(earliest_end, nobs + 1):
            latest_start = end - minimum
            best_value = -np.inf
            best_start = -1

            for start in range(earliest_start, latest_start + 1):
                prefix_value = best[segments - 1, start]
                if not np.isfinite(prefix_value):
                    continue

                _, segment_log_likelihood = _segment_fit(
                    start,
                    end,
                    cumulative_counts=cumulative_counts,
                    cumulative_log_factorials=cumulative_log_factorials,
                )
                candidate = prefix_value + segment_log_likelihood

                if candidate > best_value:
                    best_value = candidate
                    best_start = start

            best[segments, end] = best_value
            previous[segments, end] = best_start

    change_point_counts = np.arange(0, maximum + 1, dtype=np.int64)
    optimal_log_likelihood = np.empty(maximum + 1, dtype=np.float64)
    bic = np.empty(maximum + 1, dtype=np.float64)

    for change_count in range(maximum + 1):
        segments = change_count + 1
        log_likelihood = float(best[segments, nobs])
        if not np.isfinite(log_likelihood):
            raise RuntimeError("failed to find a feasible Poisson segmentation.")

        parameter_count = 2 * change_count + 1
        optimal_log_likelihood[change_count] = log_likelihood
        bic[change_count] = (
            np.log(nobs) * parameter_count - 2.0 * log_likelihood
        )

    selected_change_count = int(np.argmin(bic))
    selected_segments = selected_change_count + 1
    boundaries = _reconstruct_boundaries(
        previous,
        segments=selected_segments,
        nobs=nobs,
    )
    change_points = boundaries[1:-1]

    rates = np.empty(selected_segments, dtype=np.float64)
    for index in range(selected_segments):
        rate, _ = _segment_fit(
            int(boundaries[index]),
            int(boundaries[index + 1]),
            cumulative_counts=cumulative_counts,
            cumulative_log_factorials=cumulative_log_factorials,
        )
        rates[index] = rate

    return PoissonMultipleChangePointResult(
        change_points=change_points,
        segment_boundaries=boundaries,
        segment_rates=rates,
        selected_change_points=selected_change_count,
        selected_log_likelihood=float(
            optimal_log_likelihood[selected_change_count]
        ),
        selected_bic=float(bic[selected_change_count]),
        change_point_counts=change_point_counts,
        optimal_log_likelihood=optimal_log_likelihood,
        bic_by_change_count=bic,
    )
