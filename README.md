# Poisson Time-Series Models

A statistical Python project for modelling count-valued time series with time-varying intensity.

The original repository compared ARIMA and LSTM forecasting against simulated Poisson counts. Those experiments are retained under `legacy/` for provenance, but they are not the maintained direction of the project.

The maintained code focuses on the data-generating process itself: conditional count distributions, explicit intensity models, seasonality, overdispersion, serial dependence, calibration, and probabilistic forecast evaluation.

## Statistical scope

For a count process `Y_t`, the baseline model is

```text
Y_t | lambda_t ~ Poisson(lambda_t)
```

with a time-varying conditional intensity `lambda_t > 0`.

The project will extend this baseline to cover:

- deterministic and cyclic intensity functions;
- Poisson regression with time-varying covariates;
- negative-binomial models for overdispersion;
- autoregressive count models where serial dependence is present;
- state-space intensity models;
- proper scoring rules and predictive interval calibration;
- rolling-origin evaluation for probabilistic forecasts.

Generic sequence models are not the default modelling strategy.

## Repository layout

```text
src/poisson_time_series/    maintained typed package
tests/                      unit and statistical tests
docs/                       methodology and engineering documentation
legacy/                     historical ARIMA/LSTM notebooks and source
data/                       maintained data policy and future fixtures
.github/                    CI and repository templates
```

## Installation

The project uses Poetry.

```bash
git clone https://github.com/DiogoRibeiro7/Poisson-Processes---Time-Series.git
cd Poisson-Processes---Time-Series
poetry install
```

## Development checks

```bash
poetry run ruff check src tests
poetry run mypy src tests
poetry run pytest --cov --cov-report=term-missing
poetry run mkdocs build --strict
```

## First maintained API

The first maintained component is a reproducible periodic Poisson simulator.

```python
import numpy as np

from poisson_time_series import simulate_periodic_poisson

rng = np.random.default_rng(42)
simulation = simulate_periodic_poisson(
    rates=[2.0, 4.0, 8.0, 4.0],
    cycles=3,
    rng=rng,
)

print(simulation.counts)
print(simulation.intensity)
```

The simulator returns both the sampled counts and the exact intensity used for each observation. This makes statistical validation possible without reconstructing hidden state.

## Legacy material

The original ARIMA, SARIMAX, LSTM, and notebook experiments are stored under `legacy/`. They are preserved as historical material and should not be imported by maintained code.

## Roadmap

1. establish reproducible count-process simulation;
2. implement Poisson log-linear intensity models;
3. add cyclic/Fourier intensity components;
4. add negative-binomial models for overdispersion;
5. add residual and calibration diagnostics appropriate for counts;
6. implement rolling-origin probabilistic evaluation;
7. add state-space and autoregressive count models where justified.

## License

Apache License 2.0. See `LICENSE`.
