# Count Time Series

A statistical Python project for modelling count-valued time series with explicit probabilistic structure.

The original repository focused on simulated Poisson counts and compared ARIMA and LSTM forecasting. Those experiments are retained under `legacy/` for provenance, but they are not the maintained direction of the project.

The maintained code focuses specifically on temporal count processes: serially dependent conditional intensities, seasonality, state-space dynamics, regime changes, calibration, and probabilistic forecast evaluation. Poisson and negative-binomial GLMs are retained as baseline building blocks rather than the endpoint of the project.

## Statistical scope

Poisson models provide the baseline,

```text
Y_t | lambda_t ~ Poisson(lambda_t)
```

but the project is intentionally broader than the Poisson family. Planned and maintained components include:

- deterministic and cyclic intensity functions;
- Poisson regression with time-varying covariates;
- negative-binomial observation models for overdispersion;
- INGARCH and related observation-driven count models;
- autoregressive count models where serial dependence is present;
- state-space intensity models;
- proper scoring rules and predictive interval calibration;
- rolling-origin evaluation for probabilistic forecasts.

Generic sequence models are not the default modelling strategy.

## Repository layout

```text
src/count_time_series/      maintained typed package
tests/                      unit and statistical tests
docs/                       methodology and engineering documentation
legacy/                     historical ARIMA/LSTM notebooks and source
.github/                    CI and repository workflows
```

## Installation

The project uses Poetry.

```bash
git clone https://github.com/DiogoRibeiro7/count-time-series.git
cd count-time-series
poetry install
```

## Development checks

```bash
poetry run ruff check src/count_time_series tests
poetry run mypy src/count_time_series tests
poetry run pytest --cov --cov-report=term-missing
poetry run mkdocs build --strict
```

## Maintained API

The package currently includes reproducible count simulation, baseline Poisson and NB2 regression, Poisson and NB-INGARCH models, temporal diagnostics, probabilistic rolling-origin evaluation, and Poisson log-Gaussian latent-state modelling with Laplace filtering, approximate parameter estimation, local parameter uncertainty, and backward smoothing.

```python
import numpy as np

from count_time_series import (
    DesignMatrix,
    fit_poisson_loglinear,
    simulate_periodic_poisson,
)

simulation = simulate_periodic_poisson(
    rates=[2.0, 4.0, 8.0, 4.0],
    cycles=3,
    rng=np.random.default_rng(42),
)

design = DesignMatrix(
    values=np.ones((simulation.nobs, 1)),
    columns=("intercept",),
)

result = fit_poisson_loglinear(design, simulation.counts)
print(result.coefficients)
print(result.deviance)
```

## Project boundary

This repository is for **time-dependent count processes**. General healthcare and actuarial count-regression demonstrations live in [healthcare-count-models-lab](https://github.com/DiogoRibeiro7/healthcare-count-models-lab), while Bayesian Gaussian-process intensity estimation for Poisson point processes lives in [mbgp-poisson](https://github.com/DiogoRibeiro7/mbgp-poisson).

## Legacy material

The original ARIMA, SARIMAX, LSTM, and notebook experiments are stored under `legacy/`. They are historical material and are not imported by maintained code.

## Roadmap

1. reproducible count-process simulation;
2. Poisson log-linear intensity models;
3. cyclic and Fourier intensity components;
4. Poisson and negative-binomial INGARCH models, followed by higher-order and related observation-driven count models;
5. count-specific temporal residual and calibration diagnostics;
6. rolling-origin probabilistic evaluation with expanding or fixed-width refitting, proper log scores, and randomized-PIT calibration;
7. Poisson/NB state-space and latent-intensity models, including log-Gaussian filtering and approximate marginal-likelihood parameter estimation;
8. structural breaks and regime changes in count intensity.

## License

Apache License 2.0. See `LICENSE`.
