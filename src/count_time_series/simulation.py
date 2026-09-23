"""Simulation utilities for count-valued time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_rates(rates: ArrayLike) -> FloatArray:
    """Validate and return a one-dimensional non-negative intensity profile."""
    try:
        values: FloatArray = np.asarray(rates, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("rates must contain numeric values.") from exc

    if values.ndim != 1:
        raise ValueError("rates must be one-dimensional.")
    if values.size == 0:
        raise ValueError("rates must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("rates must contain only finite values.")
    if np.any(values < 0.0):
        raise ValueError("rates must be non-negative.")

    return values


def _validate_cycles(cycles: int) -> None:
    """Validate the number of repeated intensity cycles."""
    if isinstance(cycles, bool) or not isinstance(cycles, int):
        raise TypeError("cycles must be an integer.")
    if cycles < 1:
        raise ValueError("cycles must be at least one.")


@dataclass(frozen=True, slots=True)
class PoissonSimulation:
    """A simulated Poisson count series and its conditional intensity."""

    counts: IntArray
    intensity: FloatArray
    period: int

    def __post_init__(self) -> None:
        """Validate and freeze simulation arrays."""
        counts = np.asarray(self.counts, dtype=np.int64)
        intensity = np.asarray(self.intensity, dtype=np.float64)

        if counts.ndim != 1 or intensity.ndim != 1:
            raise ValueError("counts and intensity must be one-dimensional.")
        if counts.size == 0 or counts.size != intensity.size:
            raise ValueError("counts and intensity must have the same non-zero length.")
        if np.any(counts < 0):
            raise ValueError("counts must be non-negative.")
        if not np.all(np.isfinite(intensity)) or np.any(intensity < 0.0):
            raise ValueError("intensity must contain finite non-negative values.")
        if self.period < 1 or counts.size % self.period != 0:
            raise ValueError("period must divide the simulated series length.")

        counts_frozen = np.array(counts, dtype=np.int64, copy=True)
        intensity_frozen = np.array(intensity, dtype=np.float64, copy=True)
        counts_frozen.setflags(write=False)
        intensity_frozen.setflags(write=False)

        object.__setattr__(self, "counts", counts_frozen)
        object.__setattr__(self, "intensity", intensity_frozen)

    @property
    def nobs(self) -> int:
        """Return the number of simulated observations."""
        return int(self.counts.size)

    @property
    def cycles(self) -> int:
        """Return the number of repeated intensity cycles."""
        return self.nobs // self.period


def simulate_periodic_poisson(
    rates: ArrayLike,
    *,
    cycles: int = 1,
    rng: np.random.Generator | None = None,
) -> PoissonSimulation:
    """Simulate independent Poisson counts from a repeated intensity profile.

    Parameters
    ----------
    rates:
        One full cycle of non-negative Poisson intensities.
    cycles:
        Number of times to repeat the intensity profile.
    rng:
        Optional NumPy random generator. A fresh generator is created when
        omitted; callers requiring reproducibility should pass one explicitly.
    """
    profile = _validate_rates(rates)
    _validate_cycles(cycles)

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")

    generator = np.random.default_rng() if rng is None else rng
    intensity = np.tile(profile, cycles).astype(np.float64, copy=False)
    counts = np.asarray(generator.poisson(intensity), dtype=np.int64)

    return PoissonSimulation(
        counts=counts,
        intensity=intensity,
        period=int(profile.size),
    )
