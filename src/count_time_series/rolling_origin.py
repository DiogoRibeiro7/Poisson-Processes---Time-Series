"""Rolling-origin one-step evaluation for temporal count models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .evaluation import (
    PITCalibration,
    negative_binomial_log_score,
    pit_calibration,
    poisson_log_score,
    randomized_pit_negative_binomial,
    randomized_pit_poisson,
)
from .ingarch import fit_poisson_ingarch
from .negative_binomial import _validate_dispersion
from .negative_binomial_ingarch import fit_negative_binomial_ingarch
from .poisson_regression import IntArray, _as_counts

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


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


def _freeze_bool(values: ArrayLike) -> BoolArray:
    """Return an immutable bool copy."""
    array = np.array(values, dtype=np.bool_, copy=True)
    array.setflags(write=False)
    return array


def _validate_integer(value: int, *, name: str, minimum: int) -> int:
    """Validate an integer evaluation control."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _validate_window(
    *,
    nobs: int,
    initial_window: int,
    window_size: int | None,
) -> tuple[int, int | None]:
    """Validate rolling-origin training-window controls."""
    initial = _validate_integer(
        initial_window,
        name="initial_window",
        minimum=3,
    )
    if initial >= nobs:
        raise ValueError("initial_window must be smaller than the series length.")

    if window_size is None:
        return initial, None

    width = _validate_integer(window_size, name="window_size", minimum=3)
    if width > initial:
        raise ValueError("window_size cannot exceed initial_window.")
    return initial, width


def _training_start(origin: int, *, window_size: int | None) -> int:
    """Return the first training index for one forecast origin."""
    if window_size is None:
        return 0
    return max(0, origin - window_size)


@dataclass(frozen=True, slots=True)
class RollingOriginResult:
    """One-step probabilistic evaluation across sequential forecast origins."""

    model: str
    origins: IntArray
    training_starts: IntArray
    training_ends: IntArray
    observed_counts: IntArray
    forecast_mean: FloatArray
    log_scores: FloatArray
    pit_values: FloatArray
    converged: BoolArray
    calibration: PITCalibration
    dispersion: float | None = None

    def __post_init__(self) -> None:
        """Freeze stored result arrays and validate aligned lengths."""
        lengths = {
            int(np.asarray(self.origins).size),
            int(np.asarray(self.training_starts).size),
            int(np.asarray(self.training_ends).size),
            int(np.asarray(self.observed_counts).size),
            int(np.asarray(self.forecast_mean).size),
            int(np.asarray(self.log_scores).size),
            int(np.asarray(self.pit_values).size),
            int(np.asarray(self.converged).size),
        }
        if len(lengths) != 1:
            raise ValueError("rolling-origin result arrays must have equal length.")
        if not lengths or next(iter(lengths)) == 0:
            raise ValueError("rolling-origin result arrays must not be empty.")

        object.__setattr__(self, "origins", _freeze_int(self.origins))
        object.__setattr__(self, "training_starts", _freeze_int(self.training_starts))
        object.__setattr__(self, "training_ends", _freeze_int(self.training_ends))
        object.__setattr__(self, "observed_counts", _freeze_int(self.observed_counts))
        object.__setattr__(self, "forecast_mean", _freeze_float(self.forecast_mean))
        object.__setattr__(self, "log_scores", _freeze_float(self.log_scores))
        object.__setattr__(self, "pit_values", _freeze_float(self.pit_values))
        object.__setattr__(self, "converged", _freeze_bool(self.converged))

    @property
    def n_origins(self) -> int:
        """Return the number of evaluated forecast origins."""
        return int(self.origins.size)

    @property
    def mean_log_score(self) -> float:
        """Return the mean negative log predictive density."""
        return float(np.mean(self.log_scores))

    @property
    def convergence_rate(self) -> float:
        """Return the fraction of model fits that converged."""
        return float(np.mean(self.converged))


def rolling_origin_poisson_ingarch(
    counts: ArrayLike,
    *,
    initial_window: int,
    window_size: int | None = None,
    pit_bins: int = 10,
    rng: np.random.Generator | None = None,
    require_convergence: bool = True,
) -> RollingOriginResult:
    """Evaluate one-step Poisson INGARCH forecasts over rolling origins."""
    y = _as_counts(counts)
    initial, width = _validate_window(
        nobs=y.size,
        initial_window=initial_window,
        window_size=window_size,
    )
    generator = np.random.default_rng() if rng is None else rng
    if not isinstance(generator, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")

    origins = np.arange(initial, y.size, dtype=np.int64)
    training_starts = np.empty(origins.size, dtype=np.int64)
    means = np.empty(origins.size, dtype=np.float64)
    converged = np.empty(origins.size, dtype=np.bool_)

    for index, origin_value in enumerate(origins):
        origin = int(origin_value)
        start = _training_start(origin, window_size=width)
        training_starts[index] = start
        fit = fit_poisson_ingarch(y[start:origin])
        converged[index] = fit.converged
        if require_convergence and not fit.converged:
            raise RuntimeError(f"Poisson INGARCH did not converge at origin {origin}.")
        means[index] = float(fit.forecast_mean(1)[0])

    observed = y[origins]
    scores = poisson_log_score(observed, means)
    pit = randomized_pit_poisson(observed, means, rng=generator)

    return RollingOriginResult(
        model="poisson_ingarch",
        origins=origins,
        training_starts=training_starts,
        training_ends=origins - 1,
        observed_counts=observed,
        forecast_mean=means,
        log_scores=scores,
        pit_values=pit,
        converged=converged,
        calibration=pit_calibration(pit, bins=pit_bins),
    )


def rolling_origin_negative_binomial_ingarch(
    counts: ArrayLike,
    *,
    dispersion: float,
    initial_window: int,
    window_size: int | None = None,
    pit_bins: int = 10,
    rng: np.random.Generator | None = None,
    require_convergence: bool = True,
) -> RollingOriginResult:
    """Evaluate one-step fixed-dispersion NB-INGARCH forecasts."""
    y = _as_counts(counts)
    alpha = _validate_dispersion(dispersion)
    initial, width = _validate_window(
        nobs=y.size,
        initial_window=initial_window,
        window_size=window_size,
    )
    generator = np.random.default_rng() if rng is None else rng
    if not isinstance(generator, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")

    origins = np.arange(initial, y.size, dtype=np.int64)
    training_starts = np.empty(origins.size, dtype=np.int64)
    means = np.empty(origins.size, dtype=np.float64)
    converged = np.empty(origins.size, dtype=np.bool_)

    for index, origin_value in enumerate(origins):
        origin = int(origin_value)
        start = _training_start(origin, window_size=width)
        training_starts[index] = start
        fit = fit_negative_binomial_ingarch(
            y[start:origin],
            dispersion=alpha,
        )
        converged[index] = fit.converged
        if require_convergence and not fit.converged:
            raise RuntimeError(f"NB-INGARCH did not converge at origin {origin}.")
        means[index] = float(fit.forecast_mean(1)[0])

    observed = y[origins]
    scores = negative_binomial_log_score(
        observed,
        means,
        dispersion=alpha,
    )
    pit = randomized_pit_negative_binomial(
        observed,
        means,
        dispersion=alpha,
        rng=generator,
    )

    return RollingOriginResult(
        model="negative_binomial_ingarch",
        origins=origins,
        training_starts=training_starts,
        training_ends=origins - 1,
        observed_counts=observed,
        forecast_mean=means,
        log_scores=scores,
        pit_values=pit,
        converged=converged,
        calibration=pit_calibration(pit, bins=pit_bins),
        dispersion=alpha,
    )
