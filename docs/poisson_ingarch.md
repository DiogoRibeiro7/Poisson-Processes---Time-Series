# Poisson INGARCH(1,1)

The first explicitly temporal model in the package is a Poisson INGARCH(1,1) process.

## Conditional intensity

The observation model is

```text
Y_t | F_(t-1) ~ Poisson(lambda_t)
```

with recursion

```text
lambda_t = omega + alpha * Y_(t-1) + beta * lambda_(t-1)
```

The implementation requires

```text
omega > 0
alpha >= 0
beta >= 0
alpha + beta < 1
```

The final condition gives a stable stationary mean,

```text
E[Y_t] = omega / (1 - alpha - beta)
```

under the standard INGARCH assumptions.

## Estimation

`fit_poisson_ingarch` uses conditional maximum likelihood. The first intensity is treated as fixed and the likelihood is accumulated from the second observation onward.

Optimisation is carried out on an unconstrained parameter scale. The transformation itself enforces positivity and `alpha + beta < 1`, so invalid dynamics are not repaired by clipping after optimisation.

The fitted result reports:

- `omega`, `alpha`, and `beta`;
- the fitted conditional-intensity path;
- the fixed initial intensity;
- conditional log-likelihood;
- AIC and BIC;
- optimisation convergence status, iteration count, and message.

## Simulation

`simulate_poisson_ingarch` supports an explicit burn-in period. When no initial intensity is supplied, simulation starts at the stationary unconditional mean.

## Forecasting

The one-step forecast uses the last observed count and fitted intensity:

```text
lambda_(T+1) = omega + alpha * Y_T + beta * lambda_T
```

For later horizons the conditional mean recursion becomes

```text
E[lambda_(T+h)] = omega + (alpha + beta) * E[lambda_(T+h-1)]
```

because future counts are replaced by their conditional expectations.

## Scope

This model is deliberately temporal. Generic cross-sectional count-regression teaching and healthcare examples belong in the separate `healthcare-count-models-lab` repository, while Gaussian-process point-process intensity estimation belongs in `mbgp-poisson`.
