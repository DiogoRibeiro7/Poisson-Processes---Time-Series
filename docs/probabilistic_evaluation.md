# Probabilistic forecast evaluation

Count forecasts should be evaluated as predictive distributions, not only through point-forecast errors.

This module provides model-agnostic scoring and calibration tools for Poisson and NB2 predictive distributions.

## Log score

The package uses negative log predictive density:

```text
log score = -log P(Y = y)
```

Lower values are better.

The score is proper: in expectation, a forecaster minimizes it by reporting the predictive distribution it actually believes.

The API returns one score per observation rather than silently averaging across time. Aggregation can therefore be chosen explicitly by the caller.

## Randomized PIT for counts

For continuous predictive distributions, the probability integral transform is uniform under calibration. Counts are discrete, so the ordinary PIT is not uniform.

The package uses the randomized PIT

```text
U_t = F_t(y_t - 1) + V_t * P_t(Y_t = y_t)
```

where `V_t ~ Uniform(0, 1)`.

Under a calibrated discrete predictive distribution, randomized PIT values are Uniform(0, 1).

Randomization can be made reproducible by passing a `numpy.random.Generator`.

## PIT calibration summary

`pit_calibration` reports:

- PIT sample mean;
- PIT sample variance;
- equal-width histogram frequencies;
- Kolmogorov-Smirnov statistic against Uniform(0, 1);
- the corresponding reference p-value.

For a calibrated forecast, the ideal PIT mean is approximately `0.5` and the ideal variance is approximately `1/12`.

The KS p-value is a diagnostic, not a complete forecast-quality measure. Serial dependence among forecast errors can invalidate the usual iid reference interpretation.

## NB2 parameterisation

For mean `mu` and dispersion `phi > 0`,

```text
Var(Y) = mu + phi * mu^2
```

which corresponds to negative-binomial size

```text
r = 1 / phi
```

and success probability

```text
p = r / (r + mu)
```

The same parameterisation is used by the regression and NB-INGARCH components.

## Next step

These primitives are intentionally independent of model classes. A rolling-origin evaluator can therefore refit any supported temporal model, generate one-step predictive distributions, and compare models using the same scoring and calibration functions.
