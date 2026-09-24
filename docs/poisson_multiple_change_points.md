# Multiple Poisson change points

This model extends the single-break Poisson intensity model to an unknown number of piecewise-constant regimes.

## Exact segmentation for a fixed number of breaks

For a specified number of change points `k`, the series is divided into `k + 1` Poisson segments.

Each segment has its own maximum-likelihood intensity, equal to the sample mean within that segment.

Dynamic programming finds the globally optimal segmentation for each `k` from zero through `max_change_points`, subject to `min_segment_size`.

This is not greedy binary segmentation: every reported fixed-`k` solution is globally optimal under the piecewise-constant Poisson likelihood.

## Dynamic-programming recurrence

Let `D[s, t]` be the maximum log likelihood for observations `0:t` using exactly `s` segments.

```text
D[s, t] = max_j { D[s - 1, j] + log L(j:t) }
```

Candidate segment starts respect the minimum segment size. Segment log likelihoods are computed in constant time from cumulative counts and cumulative log-factorial terms.

The total complexity is O(K n^2), where K is the maximum number of segments considered.

## Selecting the number of breaks

For `k` change points, the BIC convention is

```text
BIC_k = -2 log L_k + (2k + 1) log n.
```

The effective parameter count is

```text
(k + 1) segment rates + k breakpoint locations = 2k + 1.
```

The selected segmentation is the BIC minimum among the globally optimal fixed-`k` fits.

## Returned structure

The result reports the selected change points, complete segment boundaries, segment intensity estimates, selected log likelihood and BIC, the optimal log likelihood for every candidate number of change points, BIC for every candidate count, and BIC improvement relative to the no-break model.

## Interpretation

BIC is used as a model-selection criterion under an explicit breakpoint-counting convention. The implementation does not attach classical p-values to selected breakpoints because breakpoint selection is a non-regular inference problem.

## Scope

This exact dynamic-programming implementation is appropriate for moderate series lengths and a bounded maximum number of breakpoints. Future work can add PELT-style pruning for longer series, breakpoint uncertainty, negative-binomial segments, or dependence within regimes.
