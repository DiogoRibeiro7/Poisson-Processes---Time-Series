"""Probabilistic scoring and calibration for count forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import kstest, nbinom, poisson  # type: ignore[import-untyped]

from .negative_binomial import _validate_dispersion
from .poisson_regression import _as_counts

FloatArray = NDArray[np.float64]


def _freeze(values: ArrayLike) -> FloatArray:
    """Return an immutable float64 copy."""
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _as_positive_mean(values: ArrayLike, *, nobs: int) -> FloatArray:
    """Validate strictly positive predictive means."""
    try:
        mean = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("mean must contain numeric values.") from exc

    if mean.ndim != 1:
        raise ValueError("mean must be one-dimensional.")
    if mean.size != nobs:
        raise ValueError("mean length must match the number of observations.")
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean must contain only finite values.")
    if np.any(mean <= 0.0):
        raise ValueError("mean must be strictly positive.")
    return mean


def _validate_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Return a valid NumPy generator."""
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")
    return np.random.default_rng() if rng is None else rng


def poisson_log_score(counts: ArrayLike, mean: ArrayLike) -> FloatArray:
    """Return per-observation negative log predictive density for Poisson forecasts."""
    y = _as_counts(counts)
    mu = _as_positive_mean(mean, nobs=y.size)
    scores = -np.asarray(poisson.logpmf(y, mu), dtype=np.float64)
    return _freeze(scores)


def negative_binomial_log_score(
    counts: ArrayLike,
    mean: ArrayLike,
    *,
    dispersion: float,
) -> FloatArray:
    """Return per-observation negative log predictive density for NB2 forecasts."""
    y = _as_counts(counts)
    mu = _as_positive_mean(mean, nobs=y.size)
    alpha = _validate_dispersion(dispersion)

    size = 1.0 / alpha
    probability = size / (size + mu)
    scores = -np.asarray(nbinom.logpmf(y, size, probability), dtype=np.float64)
    return _freeze(scores)


def randomized_pit_poisson(
    counts: ArrayLike,
    mean: ArrayLike,
    *,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Return randomized PIT values for Poisson predictive distributions."""
    y = _as_counts(counts)
    mu = _as_positive_mean(mean, nobs=y.size)
    generator = _validate_rng(rng)

    lower = np.asarray(poisson.cdf(y - 1, mu), dtype=np.float64)
    mass = np.asarray(poisson.pmf(y, mu), dtype=np.float64)
    values = lower + generator.random(y.size) * mass
    return _freeze(values)


def randomized_pit_negative_binomial(
    counts: ArrayLike,
    mean: ArrayLike,
    *,
    dispersion: float,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Return randomized PIT values for NB2 predictive distributions."""
    y = _as_counts(counts)
    mu = _as_positive_mean(mean, nobs=y.size)
    alpha = _validate_dispersion(dispersion)
    generator = _validate_rng(rng)

    size = 1.0 / alpha
    probability = size / (size + mu)
    lower = np.asarray(nbinom.cdf(y - 1, size, probability), dtype=np.float64)
    mass = np.asarray(nbinom.pmf(y, size, probability), dtype=np.float64)
    values = lower + generator.random(y.size) * mass
    return _freeze(values)


@dataclass(frozen=True, slots=True)
class PITCalibration:
    """Uniform-calibration summary for randomized PIT values."""

    values: FloatArray
    bin_edges: FloatArray
    bin_frequencies: FloatArray
    mean: float
    variance: float
    ks_statistic: float
    ks_p_value: float

    def __post_init__(self) -> None:
        """Freeze stored arrays."""
        object.__setattr__(self, "values", _freeze(self.values))
        object.__setattr__(self, "bin_edges", _freeze(self.bin_edges))
        object.__setattr__(self, "bin_frequencies", _freeze(self.bin_frequencies))

    @property
    def nobs(self) -> int:
        """Return the number of PIT values."""
        return int(self.values.size)

    @property
    def bins(self) -> int:
        """Return the number of calibration bins."""
        return int(self.bin_frequencies.size)


def pit_calibration(values: ArrayLike, *, bins: int = 10) -> PITCalibration:
    """Summarize randomized PIT values against a Uniform(0, 1) reference."""
    try:
        pit = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("values must contain numeric PIT values.") from exc

    if pit.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if pit.size < 2:
        raise ValueError("at least two PIT values are required.")
    if not np.all(np.isfinite(pit)):
        raise ValueError("PIT values must be finite.")
    if np.any((pit < 0.0) | (pit > 1.0)):
        raise ValueError("PIT values must lie in [0, 1].")

    if isinstance(bins, bool) or not isinstance(bins, int):
        raise TypeError("bins must be an integer.")
    if bins < 2:
        raise ValueError("bins must be at least two.")

    frequencies, edges = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    relative = frequencies.astype(np.float64) / pit.size
    ks_result = kstest(pit, "uniform")

    return PITCalibration(
        values=pit,
        bin_edges=np.asarray(edges, dtype=np.float64),
        bin_frequencies=relative,
        mean=float(np.mean(pit)),
        variance=float(np.var(pit, ddof=1)),
        ks_statistic=float(ks_result.statistic),
        ks_p_value=float(ks_result.pvalue),
    )
