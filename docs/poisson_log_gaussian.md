# Poisson log-Gaussian state-space model

This model introduces a latent Gaussian state for the log intensity rather than making the conditional mean a deterministic function of past observed counts.

## Latent dynamics

The latent state follows a stationary AR(1):

```text
x_t = mu + phi * (x_(t-1) - mu) + eta_t
eta_t ~ Normal(0, sigma^2)
```

with

```text
abs(phi) < 1
sigma > 0
```

The observation model is

```text
Y_t | x_t ~ Poisson(exp(x_t))
```

## Exact simulation

`simulate_poisson_log_gaussian` samples the latent Gaussian state exactly from the specified model and then samples the Poisson observation conditionally on that state.

When no initial latent state is supplied, simulation starts from the stationary Gaussian distribution with variance

```text
sigma^2 / (1 - phi^2)
```

and may discard an explicit burn-in period.

## Why filtering is approximate

The Gaussian AR(1) transition is conjugate to a Gaussian observation model, but not to a Poisson observation with log link.

Therefore this is **not** a Kalman filter.

The implementation uses a sequential Laplace approximation. At each observation it:

1. propagates the previous Gaussian approximation through the AR(1) transition;
2. combines that Gaussian prior with the Poisson log-likelihood;
3. finds the posterior mode with Newton iteration;
4. approximates the posterior locally by a Gaussian whose variance is the inverse negative Hessian.

The result stores the prior mean/variance and filtered posterior mode/variance at every time point.

## Filtered intensity

For a Gaussian approximation

```text
x_t | Y_(1:t) approximately Normal(m_t, C_t)
```

the filtered count mean is approximated by the log-normal moment

```text
E[exp(x_t) | Y_(1:t)] approximately exp(m_t + C_t / 2)
```

rather than simply `exp(m_t)`.

## Forecasting

Future latent-state moments are propagated analytically under the AR(1) transition.

The forecast count mean uses

```text
E[Y_(T+h) | Y_(1:T)] approximately exp(a_h + R_h / 2)
```

where `a_h` and `R_h` are the forecast latent-state mean and variance.

## Scope

This first state-space layer treats `mu`, `phi`, and `sigma` as fixed known parameters.

It does not yet estimate those parameters, compute a marginal likelihood, smooth latent states backward in time, or produce an exact predictive distribution. Those require a separate inference layer and should not be conflated with filtering.
