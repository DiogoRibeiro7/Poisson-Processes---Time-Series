# Poisson diagnostics

A fitted Poisson model should be checked against the assumptions that made its likelihood and standard errors meaningful.

The diagnostics layer provides residual and dispersion summaries without turning them into an automatic model-selection rule.

## Pearson residuals

For count (y_t) with fitted mean (hat{mu}_t),

```text
r_t = (y_t - mu_t) / sqrt(mu_t)
```

The sum of squared Pearson residuals gives the Pearson chi-square statistic.

The corresponding dispersion ratio is

```text
Pearson chi-square / residual degrees of freedom
```

Values materially above one can indicate overdispersion, but may also reflect mean-model misspecification, serial dependence, omitted covariates, structural changes, or outliers.

## Deviance residuals

Signed deviance residuals preserve whether an observation lies above or below its fitted mean while their squared sum reconstructs the model deviance.

The deviance dispersion ratio is

```text
deviance / residual degrees of freedom
```

It provides a second scale diagnostic for model adequacy.

## Variance-to-mean ratio

The raw sample variance-to-mean ratio is included as a descriptive property of the observed count sequence.

It is not adjusted for covariates or time-varying fitted means, so it should not be interpreted as a substitute for model-based diagnostics.

## Interpretation

No single threshold determines whether a Poisson model is acceptable. Dispersion summaries should be considered alongside residual structure, temporal dependence, covariate specification, and the scientific data-generating process.

When the conditional mean model is credible but extra-Poisson variation remains, a negative-binomial observation model becomes a natural next candidate.
