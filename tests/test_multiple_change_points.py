"""Tests for exact multiple Poisson change-point segmentation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import (
    fit_poisson_multiple_change_points,
    fit_poisson_single_change_point,
)


def test_multiple_change_points_recover_clear_three_regime_series() -> None:
    rng = np.random.default_rng(42)
    counts = np.concatenate(
        [
            rng.poisson(2.0, size=90),
            rng.poisson(9.0, size=80),
            rng.poisson(3.0, size=100),
        ]
    )

    result = fit_poisson_multiple_change_points(
        counts,
        max_change_points=4,
        min_segment_size=20,
    )

    assert result.selected_change_points == 2
    np.testing.assert_allclose(result.change_points, [90, 170], atol=5)
    np.testing.assert_allclose(result.segment_rates, [2.0, 9.0, 3.0], atol=0.8)
    assert result.bic_improvement_over_null > 0.0


def test_no_break_selected_for_constant_poisson_series() -> None:
    counts = np.random.default_rng(7).poisson(4.0, size=250)

    result = fit_poisson_multiple_change_points(
        counts,
        max_change_points=4,
        min_segment_size=20,
    )

    assert result.selected_change_points == 0
    assert result.change_points.size == 0
    assert result.n_segments == 1
    assert result.segment_rates[0] == pytest.approx(np.mean(counts))


def test_all_zero_series_selects_no_break() -> None:
    result = fit_poisson_multiple_change_points(
        np.zeros(60, dtype=np.int64),
        max_change_points=3,
        min_segment_size=10,
    )

    assert result.selected_change_points == 0
    np.testing.assert_allclose(result.segment_rates, [0.0])
    assert np.all(np.isfinite(result.optimal_log_likelihood))


def test_one_break_dynamic_program_matches_single_profile_fit() -> None:
    rng = np.random.default_rng(11)
    counts = np.concatenate(
        [rng.poisson(2.0, size=80), rng.poisson(7.0, size=80)]
    )

    single = fit_poisson_single_change_point(
        counts,
        min_segment_size=20,
    )
    multiple = fit_poisson_multiple_change_points(
        counts,
        max_change_points=1,
        min_segment_size=20,
    )

    assert multiple.selected_change_points == 1
    assert multiple.change_points[0] == single.change_point
    assert multiple.selected_log_likelihood == pytest.approx(
        single.log_likelihood_change
    )
    assert multiple.selected_bic == pytest.approx(single.bic_change)


def test_model_selection_arrays_cover_zero_through_maximum() -> None:
    counts = np.random.default_rng(5).poisson(3.0, size=80)
    result = fit_poisson_multiple_change_points(
        counts,
        max_change_points=3,
        min_segment_size=10,
    )

    np.testing.assert_array_equal(result.change_point_counts, [0, 1, 2, 3])
    assert result.optimal_log_likelihood.shape == (4,)
    assert result.bic_by_change_count.shape == (4,)
    assert not result.change_points.flags.writeable
    assert not result.segment_rates.flags.writeable
    assert not result.bic_by_change_count.flags.writeable


def test_segment_boundaries_are_consistent_with_change_points() -> None:
    rng = np.random.default_rng(19)
    counts = np.concatenate(
        [rng.poisson(1.0, size=50), rng.poisson(8.0, size=50)]
    )
    result = fit_poisson_multiple_change_points(
        counts,
        max_change_points=2,
        min_segment_size=15,
    )

    np.testing.assert_array_equal(
        result.segment_boundaries[1:-1],
        result.change_points,
    )
    assert result.segment_boundaries[0] == 0
    assert result.segment_boundaries[-1] == counts.size


@pytest.mark.parametrize("max_change_points", [0, -1])
def test_rejects_invalid_maximum_change_point_range(
    max_change_points: int,
) -> None:
    with pytest.raises(ValueError):
        fit_poisson_multiple_change_points(
            np.arange(20, dtype=np.int64),
            max_change_points=max_change_points,
            min_segment_size=5,
        )


def test_rejects_non_integer_maximum_change_points() -> None:
    with pytest.raises(TypeError):
        fit_poisson_multiple_change_points(
            np.arange(20, dtype=np.int64),
            max_change_points=cast(int, 2.5),
            min_segment_size=5,
        )


@pytest.mark.parametrize("min_segment_size", [0, -1])
def test_rejects_invalid_minimum_segment_range(
    min_segment_size: int,
) -> None:
    with pytest.raises(ValueError):
        fit_poisson_multiple_change_points(
            np.arange(20, dtype=np.int64),
            max_change_points=2,
            min_segment_size=min_segment_size,
        )


def test_rejects_infeasible_requested_change_points() -> None:
    with pytest.raises(ValueError, match="too large"):
        fit_poisson_multiple_change_points(
            np.arange(20, dtype=np.int64),
            max_change_points=4,
            min_segment_size=5,
        )
