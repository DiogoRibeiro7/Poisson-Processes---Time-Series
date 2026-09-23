# Negative-binomial INGARCH(1,1)

NB-INGARCH combines observation-driven temporal dependence with an overdispersed NB2 observation model.

## Conditional mean dynamics

The conditional mean follows the same recursion as Poisson INGARCH:

```text
mu_t = omega + alpha * Y_(t-1) + beta * mu_(t-1)
```

with

```text
omega > 0
alpha >= 0
beta >= 0
alpha + beta < 1
```

The stationary unconditional mean is therefore

```text
E[Y_t] = omega / (1 - alpha - beta)
```

under the standard assumptions.

## Conditional observation variance

The observation distribution uses the NB2 variance function

```text
Var(Y_t | F_(t-1)) = mu_t + phi * mu_t^2
```

where `phi > 0` is the dispersion parameter.

The current implementation treats `phi` as fixed and explicit. It estimates only the temporal dynamics conditional on the supplied dispersion.

## Conditional likelihood

The first intensity is fixed, and the NB2 conditional log-likelihood is accumulated from the second observation onward.

Because dispersion is fixed, AIC and BIC count three estimated parameters: `omega`, `alpha`, and `beta`.

## Simulation

`simulate_negative_binomial_ingarch` supports burn-in and stores both the conditional mean and conditional variance paths.

## Forecasting

The multi-step conditional-mean recursion is identical to Poisson INGARCH.

Only the **one-step** conditional forecast variance is exposed directly:

```text
Var(Y_(T+1) | F_T) = mu_(T+1) + phi * mu_(T+1)^2
```

For horizons beyond one step, predictive variance also includes uncertainty propagated through future random counts and is not equal to the simple NB2 conditional variance evaluated at the mean forecast. The API deliberately does not report that shortcut as a multi-step predictive variance.

## Interpretation

NB-INGARCH is useful when both of the following are present:

- serial dependence in the conditional count mean;
- conditional variation larger than the Poisson variance.

It does not replace checks for omitted seasonality, interventions, structural breaks, or latent-state dynamics.
