"""Approximate uncertainty for Poisson log-Gaussian state-space parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm  # type: ignore[import-untyped]

from .poisson_log_gaussian import _validate_positive
from .poisson_log_gaussian_estimation import (
    PoissonLogGaussianParameterFit,
    _inverse_transform,
    _transform_parameters,
    laplace_log_marginal_likelihood,
)

FloatArray = NDArray[np.float64]


def _freeze(values: ArrayLike) -> FloatArray:
    """Return an immutable float64 copy."""
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _central_hessian(
    objective: Callable[[FloatArray], float],
    theta: FloatArray,
    *,
    step: float,
) -> FloatArray:
    """Compute a central finite-difference Hessian."""
    dimension = theta.size
    hessian = np.empty((dimension, dimension), dtype=np.float64)
    base = objective(theta)

    for i in range(dimension):
        unit_i = np.zeros(dimension, dtype=np.float64)
        unit_i[i] = step
        plus = objective(theta + unit_i)
        minus = objective(theta - unit_i)
        hessian[i, i] = (plus - 2.0 * base + minus) / step**2

        for j in range(i + 1, dimension):
            unit_j = np.zeros(dimension, dtype=np.float64)
            unit_j[j] = step
            value = (
                objective(theta + unit_i + unit_j)
                - objective(theta + unit_i - unit_j)
                - objective(theta - unit_i + unit_j)
                + objective(theta - unit_i - unit_j)
            ) / (4.0 * step**2)
            hessian[i, j] = value
            hessian[j, i] = value

    return np.asarray(hessian, dtype=np.float64)


def _parameter_jacobian(theta: FloatArray) -> FloatArray:
    """Return the Jacobian from unconstrained to model parameters."""
    _, phi, state_sd = _transform_parameters(theta)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0 - phi**2, 0.0],
            [0.0, 0.0, state_sd],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class PoissonLogGaussianParameterUncertainty:
    """Approximate covariance and intervals from the Laplace objective Hessian."""

    covariance_unconstrained: FloatArray
    covariance_parameters: FloatArray
    standard_errors: FloatArray
    confidence_level: float
    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        """Validate and freeze uncertainty arrays."""
        if self.covariance_unconstrained.shape != (3, 3):
            raise ValueError("covariance_unconstrained must have shape (3, 3).")
        if self.covariance_parameters.shape != (3, 3):
            raise ValueError("covariance_parameters must have shape (3, 3).")
        if self.standard_errors.shape != (3,):
            raise ValueError("standard_errors must have length three.")
        if self.lower.shape != (3,) or self.upper.shape != (3,):
            raise ValueError("confidence interval arrays must have length three.")
        if np.any(self.standard_errors <= 0.0):
            raise ValueError("standard_errors must be strictly positive.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one.")

        object.__setattr__(
            self,
            "covariance_unconstrained",
            _freeze(self.covariance_unconstrained),
        )
        object.__setattr__(
            self,
            "covariance_parameters",
            _freeze(self.covariance_parameters),
        )
        object.__setattr__(self, "standard_errors", _freeze(self.standard_errors))
        object.__setattr__(self, "lower", _freeze(self.lower))
        object.__setattr__(self, "upper", _freeze(self.upper))

    @property
    def correlation_parameters(self) -> FloatArray:
        """Return the approximate parameter correlation matrix."""
        scale = np.outer(self.standard_errors, self.standard_errors)
        return _freeze(self.covariance_parameters / scale)


def estimate_poisson_log_gaussian_uncertainty(
    fit: PoissonLogGaussianParameterFit,
    *,
    hessian_step: float = 1e-4,
    confidence_level: float = 0.95,
    filter_tolerance: float = 1e-10,
    filter_max_iterations: int = 100,
) -> PoissonLogGaussianParameterUncertainty:
    """Estimate local parameter uncertainty from the approximate likelihood Hessian."""
    if not isinstance(fit, PoissonLogGaussianParameterFit):
        raise TypeError("fit must be a PoissonLogGaussianParameterFit.")
    if not fit.converged:
        raise ValueError("fit must have converged before uncertainty estimation.")

    step = _validate_positive(hessian_step, name="hessian_step")
    filter_tol = _validate_positive(filter_tolerance, name="filter_tolerance")
    if isinstance(filter_max_iterations, bool) or not isinstance(filter_max_iterations, int):
        raise TypeError("filter_max_iterations must be an integer.")
    if filter_max_iterations < 1:
        raise ValueError("filter_max_iterations must be at least one.")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")

    counts = fit.filter_result.observed_counts
    theta_hat = _inverse_transform(fit.state_mean, fit.phi, fit.state_sd)

    def objective(theta: FloatArray) -> float:
        state_mean, phi, state_sd = _transform_parameters(theta)
        return -laplace_log_marginal_likelihood(
            counts,
            state_mean=state_mean,
            phi=phi,
            state_sd=state_sd,
            filter_tolerance=filter_tol,
            filter_max_iterations=filter_max_iterations,
        )

    hessian = _central_hessian(objective, theta_hat, step=step)
    if not np.all(np.isfinite(hessian)):
        raise FloatingPointError("non-finite Hessian encountered.")

    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("approximate likelihood Hessian is not positive definite.")

    covariance_unconstrained = np.asarray(np.linalg.inv(hessian), dtype=np.float64)
    jacobian = _parameter_jacobian(theta_hat)
    covariance_parameters = np.asarray(
        jacobian @ covariance_unconstrained @ jacobian.T,
        dtype=np.float64,
    )
    standard_errors = np.asarray(
        np.sqrt(np.diag(covariance_parameters)),
        dtype=np.float64,
    )

    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    estimates = np.asarray(
        [fit.state_mean, fit.phi, fit.state_sd],
        dtype=np.float64,
    )
    lower = estimates - z * standard_errors
    upper = estimates + z * standard_errors

    lower[1] = max(lower[1], -1.0)
    upper[1] = min(upper[1], 1.0)
    lower[2] = max(lower[2], 0.0)

    return PoissonLogGaussianParameterUncertainty(
        covariance_unconstrained=covariance_unconstrained,
        covariance_parameters=covariance_parameters,
        standard_errors=standard_errors,
        confidence_level=float(confidence_level),
        lower=lower,
        upper=upper,
    )
