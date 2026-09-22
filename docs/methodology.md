# Methodology

For the baseline model,

```text
Y_t | lambda_t ~ Poisson(lambda_t)
```

the conditional mean and variance are both `lambda_t`.

This equality is a modelling assumption, not a property that should be forced onto every count series. Empirical overdispersion or underdispersion is evidence that a different observation model may be needed.

The first simulator repeats a user-supplied intensity profile over a fixed number of cycles. It is intentionally simple: observations are conditionally independent given the deterministic intensity sequence. Later models may introduce stochastic intensity dynamics or serial dependence explicitly.

Randomness is controlled through `numpy.random.Generator`, making reproducible simulation part of the public API.
