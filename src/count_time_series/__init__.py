"""Statistical models and tools for count-valued time series."""

from .diagnostics import PoissonDiagnostics, poisson_diagnostics
from .evaluation import (
    PITCalibration,
    negative_binomial_log_score,
    pit_calibration,
    poisson_log_score,
    randomized_pit_negative_binomial,
    randomized_pit_poisson,
)
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
from .poisson_log_gaussian import (
    PoissonLaplaceFilterResult,
    PoissonLogGaussianSimulation,
    filter_poisson_log_gaussian,
    simulate_poisson_log_gaussian,
)
from .poisson_regression import (
    DesignMatrix,
    PoissonRegressionResult,
    fit_poisson_loglinear,
)
from .rolling_origin import (
    RollingOriginResult,
    rolling_origin_negative_binomial_ingarch,
    rolling_origin_poisson_ingarch,
)
from .simulation import PoissonSimulation, simulate_periodic_poisson

__all__ = [
    "DesignMatrix",
    "INGARCHDiagnostics",
    "NegativeBinomialINGARCHResult",
    "NegativeBinomialINGARCHSimulation",
    "NegativeBinomialRegressionResult",
    "PITCalibration",
    "PoissonDiagnostics",
    "PoissonINGARCHResult",
    "PoissonINGARCHSimulation",
    "PoissonLaplaceFilterResult",
    "PoissonLogGaussianSimulation",
    "PoissonRegressionResult",
    "PoissonSimulation",
    "RollingOriginResult",
    "filter_poisson_log_gaussian",
    "fit_negative_binomial_ingarch",
    "fit_negative_binomial_loglinear",
    "fit_poisson_ingarch",
    "fit_poisson_loglinear",
    "ingarch_diagnostics",
    "negative_binomial_log_score",
    "pit_calibration",
    "poisson_diagnostics",
    "poisson_log_score",
    "randomized_pit_negative_binomial",
    "randomized_pit_poisson",
    "rolling_origin_negative_binomial_ingarch",
    "rolling_origin_poisson_ingarch",
    "simulate_negative_binomial_ingarch",
    "simulate_periodic_poisson",
    "simulate_poisson_ingarch",
    "simulate_poisson_log_gaussian",
]

__version__ = "0.1.0"
