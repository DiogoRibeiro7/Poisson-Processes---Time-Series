# Single Poisson intensity change point

The first structural-break model assumes a count series is conditionally independent Poisson with one unknown change in a piecewise-constant intensity.

## Model

For an unknown breakpoint `tau`,

```text
Y_t ~ Poisson(lambda_1),  t < tau
Y_t ~ Poisson(lambda_2),  t >= tau
```

where both segment rates are estimated from the data.

The breakpoint is restricted so each segment contains at least `min_segment_size` observations.

## Profile likelihood

For each admissible `tau`, the Poisson MLEs are the segment sample means.

The implementation evaluates the maximized log likelihood for every admissible breakpoint and selects the largest one.

Cumulative count sums and cumulative log-factorial sums make this profile evaluation linear in the series length.

## Null model

The no-break model has one constant Poisson intensity equal to the full-sample mean.

The result reports:

- the selected breakpoint;
- pre- and post-break rates;
- the no-break rate;
- change-model and null log likelihoods;
- twice the log-likelihood improvement;
- BIC for the null and change models;
- the complete profile likelihood across candidate breakpoints.

## Why there is no naive p-value

The selected breakpoint is chosen by maximizing the likelihood over many candidate locations.

Therefore the ordinary chi-square reference distribution for a likelihood-ratio test with a fixed, known breakpoint does not apply directly.

The library deliberately reports the likelihood-ratio improvement without attaching a misleading chi-square p-value.

## BIC convention

The null model is counted as one estimated parameter.

The single-break model is counted as three parameters:

- `lambda_1`;
- `lambda_2`;
- the breakpoint location.

The result exposes `bic_improvement = BIC_null - BIC_change`, so positive values favor the break model under this convention.

## Scope

This model assumes piecewise-constant independent Poisson observations.

It does not yet allow multiple breakpoints, overdispersion, autocorrelation, covariates, or formal breakpoint uncertainty.
