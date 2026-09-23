"""Tests for rolling-origin probabilistic evaluation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import (
    rolling_origin_negative_binomial_ingarch,
    rolling_origin_poisson_ingarch,
    simulate_negative_binomial_ingarch,
    simulate_poisson_ingarch,
)


def test_poisson_rolling_origin_expanding_window_alignment() -> None:
    simulation = simulate_poisson_ingarch(
        70,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        burn_in=200,
        rng=np.random.default_rng(42),
    )
    result = rolling_origin_poisson_ingarch(
        simulation.counts,
        initial_window=50,
        pit_bins=5,
        rng=np.random.default_rng(1),
    )

    assert result.n_origins == 20
    np.testing.assert_array_equal(result.origins, np.arange(50, 70))
    np.testing.assert_array_equal(result.training_starts, np.zeros(20, dtype=np.int64))
    np.testing.assert_array_equal(result.training_ends, np.arange(49, 69))
    np.testing.assert_array_equal(result.observed_counts, simulation.counts[50:])
    assert np.all(result.forecast_mean > 0.0)
    assert np.all(result.log_scores >= 0.0)
    assert np.all((result.pit_values >= 0.0) & (result.pit_values <= 1.0))
    assert result.calibration.nobs == 20
    assert result.calibration.bins == 5
    assert result.model == "poisson_ingarch"
    assert result.dispersion is None
    assert not result.origins.flags.writeable
    assert not result.log_scores.flags.writeable


def test_poisson_rolling_origin_fixed_window_alignment() -> None:
    simulation = simulate_poisson_ingarch(
        70,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        burn_in=200,
        rng=np.random.default_rng(7),
    )
    result = rolling_origin_poisson_ingarch(
        simulation.counts,
        initial_window=50,
        window_size=30,
        pit_bins=5,
        rng=np.random.default_rng(2),
    )

    np.testing.assert_array_equal(result.training_starts, np.arange(20, 40))
    assert np.all(result.origins - result.training_starts == 30)


def test_nb_rolling_origin_retains_dispersion_and_scores() -> None:
    simulation = simulate_negative_binomial_ingarch(
        70,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        dispersion=0.30,
        burn_in=200,
        rng=np.random.default_rng(11),
    )
    result = rolling_origin_negative_binomial_ingarch(
        simulation.counts,
        dispersion=0.30,
        initial_window=50,
        pit_bins=5,
        rng=np.random.default_rng(3),
    )

    assert result.model == "negative_binomial_ingarch"
    assert result.dispersion == pytest.approx(0.30)
    assert result.n_origins == 20
    assert np.all(result.forecast_mean > 0.0)
    assert np.all(np.isfinite(result.log_scores))


def test_rolling_origin_is_reproducible_with_explicit_pit_rng() -> None:
    simulation = simulate_poisson_ingarch(
        65,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        burn_in=150,
        rng=np.random.default_rng(17),
    )
    first = rolling_origin_poisson_ingarch(
        simulation.counts,
        initial_window=50,
        pit_bins=5,
        rng=np.random.default_rng(9),
    )
    second = rolling_origin_poisson_ingarch(
        simulation.counts,
        initial_window=50,
        pit_bins=5,
        rng=np.random.default_rng(9),
    )

    np.testing.assert_allclose(first.forecast_mean, second.forecast_mean)
    np.testing.assert_allclose(first.log_scores, second.log_scores)
    np.testing.assert_allclose(first.pit_values, second.pit_values)


def test_rolling_origin_summary_properties() -> None:
    simulation = simulate_poisson_ingarch(
        65,
        omega=1.0,
        alpha=0.20,
        beta=0.50,
        burn_in=150,
        rng=np.random.default_rng(22),
    )
    result = rolling_origin_poisson_ingarch(
        simulation.counts,
        initial_window=50,
        pit_bins=5,
        rng=np.random.default_rng(4),
    )

    assert result.mean_log_score == pytest.approx(float(np.mean(result.log_scores)))
    assert 0.0 <= result.convergence_rate <= 1.0


@pytest.mark.parametrize("initial_window", [0, 1, 2, 10])
def test_rolling_origin_rejects_invalid_initial_window(initial_window: int) -> None:
    counts = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError):
        rolling_origin_poisson_ingarch(
            counts,
            initial_window=initial_window,
        )


def test_rolling_origin_rejects_non_integer_initial_window() -> None:
    with pytest.raises(TypeError):
        rolling_origin_poisson_ingarch(
            np.arange(10, dtype=np.int64),
            initial_window=cast(int, 4.5),
        )


@pytest.mark.parametrize("window_size", [0, 1, 2, 6])
def test_rolling_origin_rejects_invalid_window_size(window_size: int) -> None:
    with pytest.raises(ValueError):
        rolling_origin_poisson_ingarch(
            np.arange(12, dtype=np.int64),
            initial_window=5,
            window_size=window_size,
        )


def test_rolling_origin_rejects_non_integer_window_size() -> None:
    with pytest.raises(TypeError):
        rolling_origin_poisson_ingarch(
            np.arange(12, dtype=np.int64),
            initial_window=5,
            window_size=cast(int, 4.5),
        )


def test_nb_rolling_origin_rejects_invalid_dispersion() -> None:
    with pytest.raises(ValueError):
        rolling_origin_negative_binomial_ingarch(
            np.arange(12, dtype=np.int64),
            dispersion=0.0,
            initial_window=5,
        )


def test_rolling_origin_rejects_invalid_rng() -> None:
    with pytest.raises(TypeError):
        rolling_origin_poisson_ingarch(
            np.arange(12, dtype=np.int64),
            initial_window=5,
            rng=cast(np.random.Generator, object()),
        )
