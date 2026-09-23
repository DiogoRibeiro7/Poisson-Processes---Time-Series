# Methodology

Count Time Series starts from the observation distribution and the conditional structure of the process rather than treating counts as an ordinary continuous response.

For the Poisson baseline,

```text
Y_t | lambda_t ~ Poisson(lambda_t)
```

the conditional mean and variance are both `lambda_t`.

That equality is a modelling assumption, not a property that should be imposed on every count series. Empirical overdispersion or underdispersion is evidence that another observation model may be needed.

The initial simulator repeats a user-supplied intensity profile over a fixed number of cycles. Observations are conditionally independent given that deterministic intensity sequence. Later models may introduce stochastic intensity dynamics, additional dispersion, or serial dependence explicitly.

Randomness is controlled through `numpy.random.Generator`, making reproducible simulation part of the public API.
