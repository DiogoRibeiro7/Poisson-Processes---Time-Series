"""Temporal diagnostics for fitted INGARCH count models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2  # type: ignore[import-untyped]

from .ingarch import PoissonINGARCHResult

FloatArray = NDArray[np.float64]


def _freeze(values: ArrayLike) -> FloatArray:
    """Return an immutable float64 copy."""
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _validate_lags(lags: int, *, nobs: int) -> None:
    """Validate a maximum residual-autocorrelation lag."""
    if isinstance(lags, bool) or not isinstance(lags, int):
        raise TypeError("lags must be an integer.")
    if lags < 1:
        raise ValueError("lags must be at least one.")
    if lags >= nobs:
        raise ValueError("lags must be smaller than the diagnostic sample size.")


def _validate_model_df(model_df: int, *, lags: int) -> int:
    """Validate the Ljung-Box degrees-of-freedom adjustment."""
    if isinstance(model_df, bool) or not isinstance(model_df, int):
        raise TypeError("model_df must be an integer.")
    if model_df < 0:
        raise ValueError("model_df must be non-negative.")

    degrees_of_freedom = lags - model_df
    if degrees_of_freedom <= 0:
        raise ValueError("lags - model_df must be positive.")
    return degrees_of_freedom


def _autocorrelation(values: FloatArray, *, lags: int) -> FloatArray:
    """Compute the biased sample autocorrelation through the requested lag."""
    centered = values - np.mean(values)
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise ValueError("autocorrelation is undefined for zero-variance residuals.")

    acf = np.empty(lags + 1, dtype=np.float64)
    acf[0] = 1.0

    for lag in range(1, lags + 1):
        acf[lag] = float(centered[lag:] @ centered[:-lag]) / denominator

    return _freeze(acf)


@dataclass(frozen=True, slots=True)
class INGARCHDiagnostics:
    """Temporal residual diagnostics for a fitted Poisson INGARCH model."""

    pearson_residuals: FloatArray
    autocorrelation: FloatArray
    residual_mean: float
    residual_variance: float
    ljung_box_statistic: float
    ljung_box_p_value: float
    ljung_box_lags: int
    ljung_box_degrees_of_freedom: int
    model_df: int

    def __post_init__(self) -> None:
        """Freeze residual and autocorrelation arrays."""
        object.__setattr__(self, "pearson_residuals", _freeze(self.pearson_residuals))
        object.__setattr__(self, "autocorrelation", _freeze(self.autocorrelation))

    @property
    def nobs(self) -> int:
        """Return the effective diagnostic sample size."""
        return int(self.pearson_residuals.size)


def ingarch_diagnostics(
    result: PoissonINGARCHResult,
    *,
    lags: int,
    model_df: int = 0,
) -> INGARCHDiagnostics:
    """Compute Pearson-residual autocorrelation and Ljung-Box diagnostics.

    The fixed first INGARCH observation is excluded so the diagnostic sample
    matches the conditional likelihood used for estimation.
    """
    if not isinstance(result, PoissonINGARCHResult):
        raise TypeError("result must be a PoissonINGARCHResult.")

    counts = result.observed_counts[1:].astype(np.float64, copy=False)
    intensity = result.fitted_intensity[1:]
    nobs = int(counts.size)

    if nobs < 2:
        raise ValueError("at least two conditional residuals are required.")
    if intensity.size != counts.size:
        raise ValueError("counts and fitted intensity must have matching lengths.")
    if np.any(intensity <= 0.0) or not np.all(np.isfinite(intensity)):
        raise ValueError("fitted intensity must contain finite positive values.")

    _validate_lags(lags, nobs=nobs)
    degrees_of_freedom = _validate_model_df(model_df, lags=lags)

    residuals = np.asarray(
        (counts - intensity) / np.sqrt(intensity),
        dtype=np.float64,
    )
    acf = _autocorrelation(residuals, lags=lags)

    statistic = 0.0
    for lag in range(1, lags + 1):
        statistic += (acf[lag] ** 2) / (nobs - lag)
    statistic *= nobs * (nobs + 2)

    return INGARCHDiagnostics(
        pearson_residuals=residuals,
        autocorrelation=acf,
        residual_mean=float(np.mean(residuals)),
        residual_variance=float(np.var(residuals, ddof=1)),
        ljung_box_statistic=float(statistic),
        ljung_box_p_value=float(chi2.sf(statistic, degrees_of_freedom)),
        ljung_box_lags=lags,
        ljung_box_degrees_of_freedom=degrees_of_freedom,
        model_df=model_df,
    )
