"""Tests for Poisson log-Gaussian state-space models."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import (
    filter_poisson_log_gaussian,
    simulate_poisson_log_gaussian,
)


def test_log_gaussian_simulation_is_reproducible() -> None:
    first = simulate_poisson_log_gaussian(
        100,
        state_mean=1.0,
        phi=0.8,
        state_sd=0.3,
        rng=np.random.default_rng(42),
    )
    second = simulate_poisson_log_gaussian(
        100,
        state_mean=1.0,
        phi=0.8,
        state_sd=0.3,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_allclose(first.latent_state, second.latent_state)
    np.testing.assert_allclose(first.intensity, second.intensity)
    assert not first.counts.flags.writeable
    assert not first.latent_state.flags.writeable
    assert not first.intensity.flags.writeable


def test_log_gaussian_simulation_intensity_matches_latent_state() -> None:
    simulation = simulate_poisson_log_gaussian(
        80,
        state_mean=0.5,
        phi=0.6,
        state_sd=0.2,
        burn_in=100,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_allclose(simulation.intensity, np.exp(simulation.latent_state))


def test_laplace_filter_outputs_positive_variances_and_intensities() -> None:
    simulation = simulate_poisson_log_gaussian(
        120,
        state_mean=0.8,
        phi=0.75,
        state_sd=0.25,
        burn_in=200,
        rng=np.random.default_rng(3),
    )
    result = filter_poisson_log_gaussian(
        simulation.counts,
        state_mean=0.8,
        phi=0.75,
        state_sd=0.25,
    )

    assert result.converged
    assert result.nobs == simulation.nobs
    assert np.all(result.predicted_state_variance > 0.0)
    assert np.all(result.filtered_state_variance > 0.0)
    assert np.all(result.filtered_intensity > 0.0)
    assert np.all(result.newton_iterations >= 1)
    assert not result.filtered_state_mode.flags.writeable


def test_laplace_filter_tracks_latent_state_reasonably() -> None:
    simulation = simulate_poisson_log_gaussian(
        500,
        state_mean=1.0,
        phi=0.85,
        state_sd=0.20,
        burn_in=300,
        rng=np.random.default_rng(9),
    )
    result = filter_poisson_log_gaussian(
        simulation.counts,
        state_mean=1.0,
        phi=0.85,
        state_sd=0.20,
    )

    correlation = float(
        np.corrcoef(simulation.latent_state, result.filtered_state_mode)[0, 1]
    )
    assert correlation > 0.50


def test_state_forecast_uses_ar1_moments() -> None:
    simulation = simulate_poisson_log_gaussian(
        120,
        state_mean=0.7,
        phi=0.6,
        state_sd=0.3,
        rng=np.random.default_rng(11),
    )
    result = filter_poisson_log_gaussian(
        simulation.counts,
        state_mean=0.7,
        phi=0.6,
        state_sd=0.3,
    )

    means, variances = result.forecast_state(2)
    expected_mean = 0.7 + 0.6 * (result.filtered_state_mode[-1] - 0.7)
    expected_variance = (
        0.6**2 * result.filtered_state_variance[-1] + 0.3**2
    )

    assert means[0] == pytest.approx(expected_mean)
    assert variances[0] == pytest.approx(expected_variance)
    assert means[1] == pytest.approx(0.7 + 0.6 * (means[0] - 0.7))
    assert variances[1] == pytest.approx(0.6**2 * variances[0] + 0.3**2)


def test_count_mean_forecast_uses_lognormal_moment() -> None:
    simulation = simulate_poisson_log_gaussian(
        100,
        state_mean=0.5,
        phi=0.7,
        state_sd=0.25,
        rng=np.random.default_rng(13),
    )
    result = filter_poisson_log_gaussian(
        simulation.counts,
        state_mean=0.5,
        phi=0.7,
        state_sd=0.25,
    )
    means, variances = result.forecast_state(3)
    expected = np.exp(means + 0.5 * variances)
    np.testing.assert_allclose(result.forecast_mean(3), expected)


@pytest.mark.parametrize("phi", [-1.0, 1.0, 1.2, -1.2])
def test_state_space_rejects_nonstationary_phi(phi: float) -> None:
    with pytest.raises(ValueError):
        simulate_poisson_log_gaussian(
            10,
            state_mean=0.0,
            phi=phi,
            state_sd=0.2,
        )


@pytest.mark.parametrize("state_sd", [0.0, -0.1, np.nan, np.inf])
def test_state_space_rejects_invalid_state_sd(state_sd: float) -> None:
    with pytest.raises(ValueError):
        simulate_poisson_log_gaussian(
            10,
            state_mean=0.0,
            phi=0.5,
            state_sd=state_sd,
        )


def test_state_space_rejects_non_numeric_parameters() -> None:
    with pytest.raises(TypeError):
        simulate_poisson_log_gaussian(
            10,
            state_mean=cast(float, "bad"),
            phi=0.5,
            state_sd=0.2,
        )


def test_filter_rejects_invalid_initial_variance() -> None:
    with pytest.raises(ValueError):
        filter_poisson_log_gaussian(
            [1, 2, 3],
            state_mean=0.0,
            phi=0.5,
            state_sd=0.2,
            initial_state_variance=0.0,
        )


def test_filter_rejects_invalid_newton_controls() -> None:
    with pytest.raises(ValueError):
        filter_poisson_log_gaussian(
            [1, 2, 3],
            state_mean=0.0,
            phi=0.5,
            state_sd=0.2,
            tolerance=0.0,
        )

    with pytest.raises(ValueError):
        filter_poisson_log_gaussian(
            [1, 2, 3],
            state_mean=0.0,
            phi=0.5,
            state_sd=0.2,
            max_iterations=0,
        )


def test_forecast_rejects_invalid_horizon() -> None:
    result = filter_poisson_log_gaussian(
        [1, 2, 3, 4],
        state_mean=0.0,
        phi=0.5,
        state_sd=0.2,
    )

    with pytest.raises(ValueError):
        result.forecast_mean(0)

    with pytest.raises(TypeError):
        result.forecast_mean(cast(int, 1.5))
