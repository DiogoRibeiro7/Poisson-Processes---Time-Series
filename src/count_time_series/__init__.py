"""Statistical models and tools for count-valued time series."""

from .diagnostics import PoissonDiagnostics, poisson_diagnostics
from .ingarch import (
    PoissonINGARCHResult,
    PoissonINGARCHSimulation,
    fit_poisson_ingarch,
    simulate_poisson_ingarch,
)
from .ingarch_diagnostics import INGARCHDiagnostics, ingarch_diagnostics
from .negative_binomial import (
    NegativeBinomialRegressionResult,
    fit_negative_binomial_loglinear,
)
from .negative_binomial_ingarch import (
    NegativeBinomialINGARCHResult,
    NegativeBinomialINGARCHSimulation,
    fit_negative_binomial_ingarch,
    simulate_negative_binomial_ingarch,
)
from .poisson_regression import (
    DesignMatrix,
    PoissonRegressionResult,
    fit_poisson_loglinear,
)
from .simulation import PoissonSimulation, simulate_periodic_poisson

__all__ = [
    "DesignMatrix",
    "INGARCHDiagnostics",
    "NegativeBinomialINGARCHResult",
    "NegativeBinomialINGARCHSimulation",
    "NegativeBinomialRegressionResult",
    "PoissonDiagnostics",
    "PoissonINGARCHResult",
    "PoissonINGARCHSimulation",
    "PoissonRegressionResult",
    "PoissonSimulation",
    "fit_negative_binomial_ingarch",
    "fit_negative_binomial_loglinear",
    "fit_poisson_ingarch",
    "fit_poisson_loglinear",
    "ingarch_diagnostics",
    "poisson_diagnostics",
    "simulate_negative_binomial_ingarch",
    "simulate_periodic_poisson",
    "simulate_poisson_ingarch",
]

__version__ = "0.1.0"
