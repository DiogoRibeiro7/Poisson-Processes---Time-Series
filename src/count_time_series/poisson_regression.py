"""Poisson log-linear regression for count-valued time series."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _freeze_float(values: ArrayLike) -> FloatArray:
    """Return an immutable float64 copy."""
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _as_counts(values: ArrayLike) -> IntArray:
    """Validate non-negative integer-valued counts."""
    try:
        raw = np.asarray(values)
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("counts must contain numeric values.") from exc

    if numeric.ndim != 1:
        raise ValueError("counts must be one-dimensional.")
    if numeric.size == 0:
        raise ValueError("counts must not be empty.")
    if not np.all(np.isfinite(numeric)):
        raise ValueError("counts must contain only finite values.")
    if np.any(numeric < 0.0):
        raise ValueError("counts must be non-negative.")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError("counts must be integer-valued.")

    del raw
    return np.asarray(numeric, dtype=np.int64)


def _as_exposure(values: ArrayLike | None, *, nobs: int) -> FloatArray:
    """Validate positive exposure values, defaulting to one."""
    if values is None:
        return np.ones(nobs, dtype=np.float64)

    try:
        exposure: FloatArray = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("exposure must contain numeric values.") from exc

    if exposure.ndim != 1:
        raise ValueError("exposure must be one-dimensional.")
    if exposure.size != nobs:
        raise ValueError("exposure length must match the number of observations.")
    if not np.all(np.isfinite(exposure)):
        raise ValueError("exposure must contain only finite values.")
    if np.any(exposure <= 0.0):
        raise ValueError("exposure must be strictly positive.")

    return exposure


@dataclass(frozen=True, slots=True)
class DesignMatrix:
    """Named full-rank design matrix for count regression."""

    values: FloatArray
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the design matrix."""
        array = np.asarray(self.values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("values must be two-dimensional.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("values must have at least one row and one column.")
        if not np.all(np.isfinite(array)):
            raise ValueError("values must contain only finite values.")
        if len(self.columns) != array.shape[1]:
            raise ValueError("columns must match the number of matrix columns.")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be unique.")
        if np.linalg.matrix_rank(array) < array.shape[1]:
            raise ValueError("design matrix must have full column rank.")

        object.__setattr__(self, "values", _freeze_float(array))


@dataclass(frozen=True, slots=True)
class PoissonRegressionResult:
    """Fitted Poisson log-linear regression model."""

    coefficients: FloatArray
    covariance: FloatArray
    standard_errors: FloatArray
    observed_counts: IntArray
    fitted_intensity: FloatArray
    linear_predictor: FloatArray
    columns: tuple[str, ...]
    log_likelihood: float
    deviance: float
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        """Freeze fitted arrays."""
        counts = np.array(self.observed_counts, dtype=np.int64, copy=True)
        counts.setflags(write=False)
        object.__setattr__(self, "observed_counts", counts)
        object.__setattr__(self, "coefficients", _freeze_float(self.coefficients))
        object.__setattr__(self, "covariance", _freeze_float(self.covariance))
        object.__setattr__(self, "standard_errors", _freeze_float(self.standard_errors))
        object.__setattr__(self, "fitted_intensity", _freeze_float(self.fitted_intensity))
        object.__setattr__(self, "linear_predictor", _freeze_float(self.linear_predictor))

    @property
    def nparams(self) -> int:
        """Return the number of fitted coefficients."""
        return int(self.coefficients.size)

    @property
    def nobs(self) -> int:
        """Return the number of fitted observations."""
        return int(self.fitted_intensity.size)

    def predict_mean(
        self,
        design: DesignMatrix,
        *,
        exposure: ArrayLike | None = None,
    ) -> FloatArray:
        """Predict conditional Poisson means for a compatible design."""
        if not isinstance(design, DesignMatrix):
            raise TypeError("design must be a DesignMatrix.")
        if design.columns != self.columns:
            raise ValueError("prediction design columns must exactly match fitted columns.")

        exposure_values = _as_exposure(exposure, nobs=design.values.shape[0])
        eta = np.asarray(design.values @ self.coefficients, dtype=np.float64)
        mean = exposure_values * np.exp(np.clip(eta, -700.0, 700.0))
        return _freeze_float(mean)


def _poisson_log_likelihood(counts: IntArray, mean: FloatArray) -> float:
    """Compute the Poisson log-likelihood including the factorial term."""
    total = 0.0
    for count, expected in zip(counts, mean, strict=True):
        total += float(count) * np.log(expected) - expected - lgamma(float(count) + 1.0)
    return float(total)


def _poisson_deviance(counts: IntArray, mean: FloatArray) -> float:
    """Compute Poisson deviance relative to the saturated model."""
    terms = np.empty(counts.size, dtype=np.float64)
    positive = counts > 0
    terms[positive] = (
        counts[positive] * np.log(counts[positive] / mean[positive])
        - (counts[positive] - mean[positive])
    )
    terms[~positive] = mean[~positive]
    return float(2.0 * np.sum(terms))


def fit_poisson_loglinear(
    design: DesignMatrix,
    counts: ArrayLike,
    *,
    exposure: ArrayLike | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> PoissonRegressionResult:
    """Fit a Poisson GLM with log link by Fisher scoring / IRLS."""
    if not isinstance(design, DesignMatrix):
        raise TypeError("design must be a DesignMatrix.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one.")

    y = _as_counts(counts)
    x = design.values
    nobs, nparams = x.shape
    if y.size != nobs:
        raise ValueError("counts length must match the number of design rows.")

    exposure_values = _as_exposure(exposure, nobs=nobs)
    offset = np.log(exposure_values)

    initial_mean = max(float(np.mean(y)), 0.1)
    beta = np.zeros(nparams, dtype=np.float64)
    if "intercept" in design.columns:
        beta[design.columns.index("intercept")] = np.log(initial_mean)

    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        eta = offset + x @ beta
        mean = np.exp(np.clip(eta, -700.0, 700.0))
        if not np.all(np.isfinite(mean)) or np.any(mean <= 0.0):
            raise FloatingPointError("non-finite Poisson mean encountered during fitting.")

        working_response = eta + (y - mean) / mean
        weighted_x = x * np.sqrt(mean)[:, np.newaxis]
        weighted_z = (working_response - offset) * np.sqrt(mean)

        new_beta_raw, _, rank, _ = np.linalg.lstsq(weighted_x, weighted_z, rcond=None)
        if int(rank) < nparams:
            raise ValueError("weighted design matrix became rank deficient.")

        new_beta = np.asarray(new_beta_raw, dtype=np.float64)
        step = float(np.max(np.abs(new_beta - beta)))
        beta = new_beta
        iterations = iteration

        if step <= tolerance * (1.0 + float(np.max(np.abs(beta)))):
            converged = True
            break

    eta = offset + x @ beta
    mean = np.exp(np.clip(eta, -700.0, 700.0))
    information = x.T @ (x * mean[:, np.newaxis])
    covariance = np.asarray(np.linalg.inv(information), dtype=np.float64)
    standard_errors = np.asarray(np.sqrt(np.diag(covariance)), dtype=np.float64)

    return PoissonRegressionResult(
        coefficients=beta,
        covariance=covariance,
        standard_errors=standard_errors,
        observed_counts=y,
        fitted_intensity=mean,
        linear_predictor=np.asarray(eta, dtype=np.float64),
        columns=design.columns,
        log_likelihood=_poisson_log_likelihood(y, mean),
        deviance=_poisson_deviance(y, mean),
        iterations=iterations,
        converged=converged,
    )
