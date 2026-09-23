# Count Time Series

This project studies count-valued time series through explicit probabilistic models for conditional means, intensities, dispersion, and serial structure.

The maintained code is organised around the data-generating process rather than generic forecasting wrappers.

## Current components

- reproducible periodic Poisson simulation;
- typed immutable simulation results;
- Poisson log-linear regression with exposure support;
- explicit access to fitted conditional intensities, likelihood, deviance, and model-based uncertainty.

## Planned components

- cyclic intensity bases;
- negative-binomial models;
- diagnostics for dispersion, count residuals, and calibration;
- probabilistic rolling-origin evaluation;
- state-space and autoregressive count models.
