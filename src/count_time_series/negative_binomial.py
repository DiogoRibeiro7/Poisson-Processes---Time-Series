"""Negative-binomial regression for overdispersed count data."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np
from numpy.typing import ArrayLike

from .poisson_regression import (
    DesignMatrix,
    FloatArray,
    IntArray,
    _as_counts,
    _as_exposure,
    _freeze_float,
)


@dataclass(frozen=True, slots=True)
class NegativeBinomialRegressionResult:
    """Fitted NB2 log-linear regression with fixed dispersion."""

    coefficients: FloatArray
    covariance: FloatArray
    standard_errors: FloatArray
    observed_counts: IntArray
    fitted_mean: FloatArray
    fitted_variance: FloatArray
    linear_predictor: FloatArray
    columns: tuple[str, ...]
    dispersion: float
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
        object.__setattr__(self, "fitted_mean", _freeze_float(self.fitted_mean))
        object.__setattr__(self, "fitted_variance", _freeze_float(self.fitted_variance))
        object.__setattr__(self, "linear_predictor", _freeze_float(self.linear_predictor))

    @property
    def nparams(self) -> int:
        """Return the number of fitted regression coefficients."""
        return int(self.coefficients.size)

    @property
    def nobs(self) -> int:
        """Return the number of fitted observations."""
        return int(self.fitted_mean.size)

    def predict_mean(
        self,
        design: DesignMatrix,
        *,
        exposure: ArrayLike | None = None,
    ) -> FloatArray:
        """Predict conditional means for a compatible named design."""
        if not isinstance(design, DesignMatrix):
            raise TypeError("design must be a DesignMatrix.")
        if design.columns != self.columns:
            raise ValueError("prediction design columns must exactly match fitted columns.")

        exposure_values = _as_exposure(exposure, nobs=design.values.shape[0])
        eta = np.asarray(design.values @ self.coefficients, dtype=np.float64)
        mean = exposure_values * np.exp(np.clip(eta, -700.0, 700.0))
        return _freeze_float(mean)

    def predict_variance(
        self,
        design: DesignMatrix,
        *,
        exposure: ArrayLike | None = None,
    ) -> FloatArray:
        """Predict the NB2 conditional variance."""
        mean = self.predict_mean(design, exposure=exposure)
        variance = mean + self.dispersion * mean**2
        return _freeze_float(variance)


def _validate_dispersion(dispersion: float) -> float:
    """Validate the NB2 dispersion parameter alpha."""
    if isinstance(dispersion, bool) or not isinstance(dispersion, (int, float)):
        raise TypeError("dispersion must be a positive finite float.")

    value = float(dispersion)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("dispersion must be finite and strictly positive.")
    return value


def _negative_binomial_log_likelihood(
    counts: IntArray,
    mean: FloatArray,
    *,
    dispersion: float,
) -> float:
    """Compute the NB2 log-likelihood for fixed dispersion."""
    size = 1.0 / dispersion
    total = 0.0

    for count, expected in zip(counts, mean, strict=True):
        y = float(count)
        denominator = size + expected
        total += (
            lgamma(y + size)
            - lgamma(size)
            - lgamma(y + 1.0)
            + size * (np.log(size) - np.log(denominator))
            + y * (np.log(expected) - np.log(denominator))
        )

    return float(total)


def _negative_binomial_deviance(
    counts: IntArray,
    mean: FloatArray,
    *,
    dispersion: float,
) -> float:
    """Compute NB2 deviance relative to the saturated model."""
    size = 1.0 / dispersion
    y = counts.astype(np.float64, copy=False)

    first = np.zeros(y.size, dtype=np.float64)
    positive = y > 0.0
    first[positive] = y[positive] * np.log(y[positive] / mean[positive])

    second = (y + size) * np.log((y + size) / (mean + size))
    return float(2.0 * np.sum(first - second))


def fit_negative_binomial_loglinear(
    design: DesignMatrix,
    counts: ArrayLike,
    *,
    dispersion: float,
    exposure: ArrayLike | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> NegativeBinomialRegressionResult:
    """Fit an NB2 log-linear model by IRLS for a fixed dispersion parameter.

    The conditional variance is mu + alpha * mu^2. The supplied dispersion
    parameter is alpha and is not estimated by this function.
    """
    if not isinstance(design, DesignMatrix):
        raise TypeError("design must be a DesignMatrix.")
    alpha = _validate_dispersion(dispersion)
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
            raise FloatingPointError(
                "non-finite negative-binomial mean encountered during fitting."
            )

        weights = mean / (1.0 + alpha * mean)
        working_response = eta + (y - mean) / mean
        sqrt_weights = np.sqrt(weights)
        weighted_x = x * sqrt_weights[:, np.newaxis]
        weighted_z = (working_response - offset) * sqrt_weights

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
    variance = mean + alpha * mean**2
    weights = mean / (1.0 + alpha * mean)
    information = x.T @ (x * weights[:, np.newaxis])
    covariance = np.asarray(np.linalg.inv(information), dtype=np.float64)
    standard_errors = np.asarray(np.sqrt(np.diag(covariance)), dtype=np.float64)

    return NegativeBinomialRegressionResult(
        coefficients=beta,
        covariance=covariance,
        standard_errors=standard_errors,
        observed_counts=y,
        fitted_mean=mean,
        fitted_variance=variance,
        linear_predictor=np.asarray(eta, dtype=np.float64),
        columns=design.columns,
        dispersion=alpha,
        log_likelihood=_negative_binomial_log_likelihood(
            y,
            mean,
            dispersion=alpha,
        ),
        deviance=_negative_binomial_deviance(
            y,
            mean,
            dispersion=alpha,
        ),
        iterations=iterations,
        converged=converged,
    )
