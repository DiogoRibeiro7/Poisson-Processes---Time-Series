"""Diagnostics for fitted Poisson count models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .poisson_regression import PoissonRegressionResult

FloatArray = NDArray[np.float64]


def _freeze(values: ArrayLike) -> FloatArray:
    """Return an immutable float64 copy."""
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class PoissonDiagnostics:
    """Residual and dispersion summaries for a fitted Poisson model."""

    pearson_residuals: FloatArray
    deviance_residuals: FloatArray
    pearson_chi_square: float
    residual_degrees_of_freedom: int
    pearson_dispersion: float
    deviance_dispersion: float
    variance_to_mean_ratio: float

    def __post_init__(self) -> None:
        """Freeze diagnostic residual arrays."""
        object.__setattr__(self, "pearson_residuals", _freeze(self.pearson_residuals))
        object.__setattr__(self, "deviance_residuals", _freeze(self.deviance_residuals))


def poisson_diagnostics(result: PoissonRegressionResult) -> PoissonDiagnostics:
    """Compute residual and dispersion diagnostics for a fitted Poisson model.

    Dispersion ratios near one are consistent with the Poisson mean-variance
    relationship, but they are descriptive diagnostics rather than binary tests.
    """
    if not isinstance(result, PoissonRegressionResult):
        raise TypeError("result must be a PoissonRegressionResult.")

    counts = result.observed_counts.astype(np.float64, copy=False)
    mean = result.fitted_intensity

    if counts.size != mean.size:
        raise ValueError("observed counts and fitted intensity must have equal length.")
    if np.any(mean <= 0.0) or not np.all(np.isfinite(mean)):
        raise ValueError("fitted intensity must contain finite positive values.")

    dof_resid = result.nobs - result.nparams
    if dof_resid <= 0:
        raise ValueError("positive residual degrees of freedom are required.")

    pearson_residuals = (counts - mean) / np.sqrt(mean)
    pearson_chi_square = float(pearson_residuals @ pearson_residuals)

    deviance_components = np.empty(counts.size, dtype=np.float64)
    positive = counts > 0.0
    deviance_components[positive] = 2.0 * (
        counts[positive] * np.log(counts[positive] / mean[positive])
        - (counts[positive] - mean[positive])
    )
    deviance_components[~positive] = 2.0 * mean[~positive]

    deviance_residuals = np.sign(counts - mean) * np.sqrt(
        np.maximum(deviance_components, 0.0)
    )

    sample_mean = float(np.mean(counts))
    if sample_mean == 0.0:
        variance_to_mean = 0.0
    elif counts.size < 2:
        variance_to_mean = 0.0
    else:
        variance_to_mean = float(np.var(counts, ddof=1) / sample_mean)

    return PoissonDiagnostics(
        pearson_residuals=pearson_residuals,
        deviance_residuals=deviance_residuals,
        pearson_chi_square=pearson_chi_square,
        residual_degrees_of_freedom=dof_resid,
        pearson_dispersion=pearson_chi_square / dof_resid,
        deviance_dispersion=result.deviance / dof_resid,
        variance_to_mean_ratio=variance_to_mean,
    )
