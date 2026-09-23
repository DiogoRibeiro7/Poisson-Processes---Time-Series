# INGARCH temporal diagnostics

Temporal count models should remove systematic serial structure from the conditional mean, not merely fit the marginal count distribution.

The INGARCH diagnostics therefore focus on standardized conditional residuals and their temporal dependence.

## Diagnostic sample

Poisson INGARCH fitting uses a fixed initial intensity and accumulates the conditional likelihood from the second observation onward.

The diagnostics use the same effective sample:

```text
t = 2, ..., n
```

The initialization point is not treated as an ordinary fitted residual.

## Pearson residuals

For fitted conditional intensity `lambda_t`,

```text
r_t = (Y_t - lambda_t) / sqrt(lambda_t)
```

Under a well-specified Poisson conditional model, these residuals should be centered near zero with variance near one, although finite-sample deviations are expected.

## Residual autocorrelation

The sample autocorrelation function is computed on the Pearson residuals after subtracting their sample mean.

Persistent residual autocorrelation indicates that the conditional-intensity recursion has not captured all temporal dependence.

## Ljung-Box diagnostic

The Ljung-Box statistic is

```text
Q = n(n + 2) * sum(rho_k^2 / (n - k))
```

through a user-selected maximum lag.

The optional `model_df` argument adjusts the chi-square reference degrees of freedom:

```text
df = lags - model_df
```

The adjustment is explicit. For a fitted INGARCH(1,1), a user may choose `model_df=3` to account for `omega`, `alpha`, and `beta`, but the library does not silently impose that convention.

## Interpretation

A small Ljung-Box p-value is evidence of residual serial dependence under the chosen reference approximation. It is not a complete model-adequacy verdict.

Residual variance materially above one may reflect overdispersion. Serial structure can instead indicate missing lags, seasonality, interventions, regime changes, or an inappropriate observation-driven specification.
