# Rolling-origin evaluation

Rolling-origin evaluation measures genuine out-of-sample one-step forecast performance while respecting time order.

## Forecast origins

For a series of length `n`, `initial_window` identifies the first forecast origin. At each origin, the model is refit using only observations available before the target count.

For origin `t`:

```text
training data = Y[start:t]
target        = Y[t]
```

The origin itself is never included in its training sample.

## Expanding and fixed-width windows

With

```text
window_size = None
```

the training set expands from the beginning of the series.

With a positive fixed `window_size`, the evaluator uses a sliding window ending immediately before the forecast origin.

The fixed window cannot exceed `initial_window`, so the first origin always has enough observations to fit the temporal model.

## Supported models

The first rolling-origin layer supports:

- Poisson INGARCH(1,1);
- fixed-dispersion NB2 INGARCH(1,1).

Both are refit independently at every forecast origin.

## Stored results

Each backtest stores:

- forecast-origin indices;
- training start and end indices;
- observed target counts;
- one-step forecast means;
- per-origin negative log predictive density;
- randomized PIT values;
- per-origin convergence flags;
- aggregate randomized-PIT calibration;
- mean log score and convergence rate.

NB-INGARCH results also retain the fixed dispersion used across origins.

## Convergence

By default, a non-converged fit stops the evaluation with an explicit error. This avoids silently mixing valid and invalid model fits into a single aggregate score.

Set `require_convergence=False` to retain and score those origins while inspecting the convergence flags explicitly.

## Randomized PIT

The backtest accepts an optional NumPy random generator. Passing one makes discrete PIT randomization reproducible across repeated evaluations.

## Interpretation

Rolling-origin scores answer a different question from in-sample likelihood, AIC, or BIC. They measure predictive performance on counts that were not used to fit the corresponding model instance.

The current implementation is one-step only. Multi-step predictive evaluation requires forecast distributions that propagate future-count uncertainty through the temporal recursion and should not be approximated by repeatedly plugging in conditional means.
