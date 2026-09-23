# Log-Gaussian parameter estimation

The Poisson log-Gaussian state-space filter treats the latent-state parameters as fixed. This inference layer estimates those parameters separately by maximizing a sequential Laplace approximation to the marginal likelihood.

## Parameterization

The model is

```text
x_t = mu + phi * (x_(t-1) - mu) + eta_t
eta_t ~ Normal(0, sigma^2)

Y_t | x_t ~ Poisson(exp(x_t))
```

Optimization is performed on an unconstrained scale:

```text
mu    = theta_1
phi   = tanh(theta_2)
sigma = exp(theta_3)
```

so `abs(phi) < 1` and `sigma > 0` hold by construction.

## Approximate marginal likelihood

At each time point the filter supplies a Gaussian predictive prior

```text
x_t | Y_(1:t-1) approximately Normal(a_t, R_t)
```

and a Laplace posterior mode `m_t`.

The predictive probability

```text
p(Y_t | Y_(1:t-1))
```

requires integrating a Poisson likelihood against the Gaussian prior. That integral is not available in closed form.

The implementation applies a one-dimensional Laplace approximation around `m_t` and sums the resulting log predictive densities across time.

The reported objective is therefore named `approximate_log_likelihood`, and the information criteria are named `approximate_aic` and `approximate_bic`.

## Separation from filtering

Two public functions are provided:

- `laplace_log_marginal_likelihood` evaluates a fixed parameter triple;
- `fit_poisson_log_gaussian_parameters` optimizes the approximate marginal likelihood and returns the corresponding filtered state trajectory.

This separation makes the approximation directly inspectable and testable.

## Initialization

When no initial value for `mu` is supplied, the optimizer starts from the logarithm of the sample mean count.

Initial values for `phi` and `sigma` are explicit arguments and are validated before optimization.

## Interpretation

The estimator is an approximate maximum-marginal-likelihood procedure, not an exact likelihood estimator.

Its uncertainty is not yet quantified. The current result reports point estimates, optimization convergence, the approximate likelihood and information criteria, and the fitted Laplace filter.

Future work can add observed-Hessian uncertainty, multiple starts, smoothing, or particle-based likelihood estimation.
