# Poisson log-linear regression

The first maintained estimator is a Poisson generalized linear model with a log link.

## Conditional mean model

For observation (t),

```text
Y_t | x_t ~ Poisson(mu_t)

log(mu_t) = log(exposure_t) + x_t^T beta
```

The exposure term is optional and defaults to one. When exposure is supplied, its logarithm enters as a fixed offset rather than an estimated coefficient.

## Estimation

The coefficient vector is estimated by Fisher scoring / iteratively reweighted least squares (IRLS). At each iteration, the implementation constructs the standard Poisson working response and weights, then solves the weighted least-squares problem with `numpy.linalg.lstsq`.

The algorithm stops when the maximum absolute coefficient change is smaller than

```text
tolerance * (1 + max(abs(beta)))
```

or when `max_iterations` is reached.

A non-converged fit is returned with `converged=False`; it is not silently presented as converged.

## Inference

The covariance matrix is based on the inverse observed Fisher information,

```text
(X' W X)^-1
```

where `W` contains the fitted Poisson means.

These standard errors are model-based. They assume the Poisson mean-variance relationship is appropriate. Overdispersion or serial dependence requires additional modelling or robust inference.

## Likelihood and deviance

The fitted result reports the full Poisson log-likelihood, including the factorial term, and the Poisson deviance relative to the saturated model.

For positive counts, the deviance contribution is

```text
2 * [ y * log(y / mu) - (y - mu) ]
```

and for zero counts it is

```text
2 * mu
```

## Example

```python
import numpy as np

from count_time_series import DesignMatrix, fit_poisson_loglinear

time = np.arange(24, dtype=float)
design = DesignMatrix(
    values=np.column_stack(
        [
            np.ones_like(time),
            np.sin(2.0 * np.pi * time / 24.0),
            np.cos(2.0 * np.pi * time / 24.0),
        ]
    ),
    columns=("intercept", "daily_sin", "daily_cos"),
)

counts = [2, 1, 3, 4, 6, 8, 10, 12, 15, 14, 12, 9,
          8, 7, 6, 5, 5, 6, 8, 10, 9, 7, 5, 3]

result = fit_poisson_loglinear(design, counts)

print(result.coefficients)
print(result.standard_errors)
print(result.deviance)
print(result.converged)
```

The design matrix is explicit. The model does not create trend, seasonality, or an intercept automatically.

## Exposure

Exposure is useful when counts arise from unequal observation windows, populations, distances, or other opportunities for events to occur.

For example, with exposure (E_t),

```text
mu_t = E_t * exp(x_t^T beta)
```

so the coefficients describe the rate per unit exposure.
