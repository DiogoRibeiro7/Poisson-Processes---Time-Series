"""Statistical models and tools for count-valued time series."""

from .poisson_regression import (
    DesignMatrix,
    PoissonRegressionResult,
    fit_poisson_loglinear,
)
from .simulation import PoissonSimulation, simulate_periodic_poisson

__all__ = [
    "DesignMatrix",
    "PoissonRegressionResult",
    "PoissonSimulation",
    "fit_poisson_loglinear",
    "simulate_periodic_poisson",
]

__version__ = "0.1.0"
