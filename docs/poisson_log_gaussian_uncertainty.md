# Log-Gaussian parameter uncertainty

The log-Gaussian state-space parameter estimator maximizes an approximate marginal likelihood. This layer derives local uncertainty from the curvature of that same approximate objective.

## Hessian scale

Optimization is performed on the unconstrained parameter vector

```text
theta = (mu, atanh(phi), log(sigma)).
```

The uncertainty routine computes a central finite-difference Hessian of the approximate negative log likelihood on this unconstrained scale.

If that Hessian is positive definite, its inverse is used as the local covariance approximation.

## Delta method

The model parameters are

```text
mu    = theta_1
phi   = tanh(theta_2)
sigma = exp(theta_3).
```

The Jacobian is

```text
diag(1, 1 - phi^2, sigma).
```

The covariance on the model-parameter scale is

```text
J * Cov(theta) * J'
```

and standard errors are the square roots of its diagonal.

## Confidence intervals

The implementation reports symmetric normal-reference intervals on the model-parameter scale.

For interpretability, interval endpoints are clipped to the natural boundaries for `phi` and `sigma`:

- `phi` remains within [-1, 1];
- the lower endpoint for `sigma` is no smaller than zero.

These are approximate local intervals, not exact frequentist confidence intervals.

## Failure modes

Uncertainty estimation raises explicitly when the finite-difference Hessian is non-finite or not positive definite.

A singular or indefinite Hessian can indicate weak local identification, numerical instability, or a poor quadratic approximation. It is therefore not silently replaced by a pseudo-inverse.

## Interpretation

The covariance, standard errors, correlations, and intervals inherit both approximations already present in the estimator:

1. the sequential Laplace approximation to the marginal likelihood;
2. the local quadratic approximation represented by the Hessian.

They should be described as approximate parameter uncertainty.
