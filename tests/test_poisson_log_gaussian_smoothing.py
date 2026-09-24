"""Tests for Poisson log-Gaussian backward smoothing."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import (
    PoissonLaplaceFilterResult,
    filter_poisson_log_gaussian,
    simulate_poisson_log_gaussian,
    smooth_poisson_log_gaussian,
)


def _example() -> tuple[np.ndarray, np.ndarray]:
    simulation = simulate_poisson_log_gaussian(
        300,
        state_mean=0.8,
        phi=0.85,
        state_sd=0.20,
        burn_in=250,
        rng=np.random.default_rng(42),
    )
    return simulation.counts, simulation.latent_state


def test_smoother_preserves_last_filtered_state() -> None:
    counts, _ = _example()
    filtered = filter_poisson_log_gaussian(
        counts,
        state_mean=0.8,
        phi=0.85,
        state_sd=0.20,
    )
    smoothed = smooth_poisson_log_gaussian(filtered)

    assert smoothed.smoothed_state_mean[-1] == pytest.approx(
        filtered.filtered_state_mode[-1]
    )
    assert smoothed.smoothed_state_variance[-1] == pytest.approx(
        filtered.filtered_state_variance[-1]
    )
    assert smoothed.smoothing_gain[-1] == pytest.approx(0.0)


def test_smoother_outputs_positive_variances_and_intensities() -> None:
    counts, _ = _example()
    smoothed = smooth_poisson_log_gaussian(
        filter_poisson_log_gaussian(
            counts,
            state_mean=0.8,
            phi=0.85,
            state_sd=0.20,
        )
    )

    assert smoothed.nobs == counts.size
    assert np.all(smoothed.smoothed_state_variance > 0.0)
    assert np.all(smoothed.smoothed_intensity > 0.0)
    assert not smoothed.smoothed_state_mean.flags.writeable
    assert not smoothed.smoothed_state_variance.flags.writeable
    assert not smoothed.smoothed_intensity.flags.writeable


def test_smoothing_reduces_average_state_variance() -> None:
    counts, _ = _example()
    filtered = filter_poisson_log_gaussian(
        counts,
        state_mean=0.8,
        phi=0.85,
        state_sd=0.20,
    )
    smoothed = smooth_poisson_log_gaussian(filtered)

    assert float(np.mean(smoothed.smoothed_state_variance)) <= float(
        np.mean(filtered.filtered_state_variance)
    )


def test_smoothing_improves_latent_state_tracking_on_example() -> None:
    counts, latent = _example()
    filtered = filter_poisson_log_gaussian(
        counts,
        state_mean=0.8,
        phi=0.85,
        state_sd=0.20,
    )
    smoothed = smooth_poisson_log_gaussian(filtered)

    filtered_rmse = float(
        np.sqrt(np.mean((filtered.filtered_state_mode - latent) ** 2))
    )
    smoothed_rmse = float(
        np.sqrt(np.mean((smoothed.smoothed_state_mean - latent) ** 2))
    )

    assert smoothed_rmse <= filtered_rmse


def test_smoothed_intensity_uses_log_normal_moment() -> None:
    counts, _ = _example()
    smoothed = smooth_poisson_log_gaussian(
        filter_poisson_log_gaussian(
            counts,
            state_mean=0.8,
            phi=0.85,
            state_sd=0.20,
        )
    )

    expected = np.exp(
        smoothed.smoothed_state_mean
        + 0.5 * smoothed.smoothed_state_variance
    )
    np.testing.assert_allclose(smoothed.smoothed_intensity, expected)


def test_smoother_rejects_non_filter_result() -> None:
    with pytest.raises(TypeError, match="PoissonLaplaceFilterResult"):
        smooth_poisson_log_gaussian(
            cast(PoissonLaplaceFilterResult, object())
        )

