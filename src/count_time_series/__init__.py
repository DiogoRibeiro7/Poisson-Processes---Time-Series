"""Statistical models and tools for count-valued time series."""

from .diagnostics import PoissonDiagnostics, poisson_diagnostics
from .negative_binomial import (
    NegativeBinomialRegressionResult,
    fit_negative_binomial_loglinear,
)
from .poisson_regression import (
    DesignMatrix,
    PoissonRegressionResult,
    fit_poisson_loglinear,
)
from .simulation import PoissonSimulation, simulate_periodic_poisson

__all__ = [
    "DesignMatrix",
    "NegativeBinomialRegressionResult",
    "PoissonDiagnostics",
    "PoissonRegressionResult",
    "PoissonSimulation",
    "fit_negative_binomial_loglinear",
    "fit_poisson_loglinear",
    "poisson_diagnostics",
    "simulate_periodic_poisson",
]

__version__ = "0.1.0"
