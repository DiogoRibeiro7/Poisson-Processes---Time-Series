# Negative-binomial regression

The project uses the NB2 parameterisation for overdispersed count data.

## Conditional distribution

For conditional mean `mu_t` and dispersion `alpha > 0`,

```text
E[Y_t | x_t]   = mu_t
Var[Y_t | x_t] = mu_t + alpha * mu_t^2
```

The mean model uses the same log link and exposure semantics as Poisson regression:

```text
log(mu_t) = log(exposure_t) + x_t^T beta
```

## Dispersion is explicit

`fit_negative_binomial_loglinear` requires the dispersion parameter to be supplied explicitly.

This is intentional. The function estimates the regression coefficients conditional on `alpha`; it does not silently substitute a method-of-moments estimate or label a tuning constant as an estimated parameter.

A separate profiling or joint-estimation layer can be added later.

## IRLS

With NB2 variance

```text
V(mu) = mu + alpha * mu^2
```

and a log link, the IRLS weight is

```text
w_t = mu_t / (1 + alpha * mu_t)
```

The working response has the same log-link form as in the Poisson model.

## Likelihood and deviance

The fitted result reports the full negative-binomial log-likelihood and the NB2 deviance relative to the saturated model.

The model-based covariance matrix is the inverse weighted information matrix,

```text
(X' W X)^-1
```

conditional on the supplied dispersion.

## Prediction

The fitted result exposes both `predict_mean` and `predict_variance`, with

```text
predicted variance = predicted mean + alpha * predicted mean^2
```

## Interpretation

Negative-binomial regression is appropriate when the conditional variance is systematically larger than the conditional mean and the NB2 quadratic variance form is scientifically plausible.

It is not a generic repair for every failed Poisson model. Serial dependence, omitted seasonality, structural change, zero inflation, and other forms of misspecification should be investigated separately.
