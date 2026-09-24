# Log-Gaussian backward smoothing

Filtering estimates the latent log intensity at time `t` using observations only through time `t`.

Smoothing uses the full observed series.

## Forward approximation

The Poisson observation model is non-Gaussian, so the forward pass is the sequential Laplace-Gaussian filter documented elsewhere.

At each time point it produces an approximate Gaussian posterior

```text
x_t | Y_(1:t) approximately Normal(m_t, C_t)
```

and a one-step predictive Gaussian

```text
x_(t+1) | Y_(1:t) approximately Normal(a_(t+1), R_(t+1)).
```

## Backward recursion

Because the state transition itself is linear Gaussian, the standard Rauch-Tung-Striebel recursion can be applied to these approximate Gaussian moments.

The smoothing gain is

```text
J_t = C_t * phi / R_(t+1)
```

and the smoothed moments are

```text
m_t^s = m_t + J_t * (m_(t+1)^s - a_(t+1))

C_t^s = C_t + J_t^2 * (C_(t+1)^s - R_(t+1)).
```

The final smoothed state equals the final filtered state because there are no future observations beyond the last time point.

## Smoothed intensity

For the Gaussian smoothed approximation

```text
x_t | Y_(1:T) approximately Normal(m_t^s, C_t^s)
```

the smoothed count mean is

```text
E[exp(x_t) | Y_(1:T)] approximately exp(m_t^s + C_t^s / 2).
```

## Interpretation

This is an RTS-style smoother applied to a Laplace-Gaussian forward approximation.

The backward recursion is analytic conditional on the forward Gaussian approximations, but the overall procedure remains approximate because the original Poisson filtering step is approximate.

Smoothing is intended for retrospective latent-state reconstruction. It must not be used as a forecasting procedure because it incorporates future observations.
