# Poisson Time-Series Models

This project studies count-valued time series through explicit probability models for the conditional intensity.

The maintained code is organised around the data-generating process rather than generic forecasting wrappers.

## Current components

- reproducible periodic Poisson simulation;
- typed immutable simulation results;
- explicit access to the latent conditional intensity used for simulation.

## Planned components

- Poisson log-linear intensity models;
- cyclic intensity bases;
- negative-binomial models;
- diagnostics for count residuals and calibration;
- probabilistic rolling-origin evaluation;
- state-space and autoregressive count models.
