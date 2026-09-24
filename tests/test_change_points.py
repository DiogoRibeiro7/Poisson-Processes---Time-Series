"""Tests for single Poisson intensity change-point estimation."""

from __future__ import annotations

import numpy as np
import pytest

from count_time_series import fit_poisson_single_change_point


def test_change_point_recovers_clear_rate_shift() -> None:
    rng = np.random.default_rng(42)
    counts = np.concatenate(
        [
            rng.poisson(2.0, size=120),
            rng.poisson(8.0, size=100),
        ]
    )

    result = fit_poisson_single_change_point(
        counts,
        min_segment_size=20,
    )

    assert result.change_point == pytest.approx(120, abs=5)
    assert result.rate_before == pytest.approx(2.0, abs=0.4)
    assert result.rate_after == pytest.approx(8.0, abs=0.8)
    assert result.likelihood_ratio > 0.0
    assert result.bic_improvement > 0.0


def test_profile_maximum_matches_selected_change_point() -> None:
    counts = np.array([1, 1, 2, 1, 8, 9, 7, 8], dtype=np.int64)
    result = fit_poisson_single_change_point(
        counts,
        min_segment_size=2,
    )

    selected = int(np.argmax(result.profile_log_likelihood))
    assert result.candidate_change_points[selected] == result.change_point
    assert not result.candidate_change_points.flags.writeable
    assert not result.profile_log_likelihood.flags.writeable


def test_null_rate_matches_sample_mean() -> None:
    counts = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    result = fit_poisson_single_change_point(
        counts,
        min_segment_size=2,
    )

    assert result.null_rate == pytest.approx(float(np.mean(counts)))


def test_zero_count_segment_is_supported() -> None:
    counts = np.array([0, 0, 0, 0, 4, 5, 6, 5], dtype=np.int64)
    result = fit_poisson_single_change_point(
        counts,
        min_segment_size=2,
    )

    assert result.rate_before == pytest.approx(0.0)
    assert result.rate_after > 0.0


def test_constant_zero_series_has_finite_profile() -> None:
    counts = np.zeros(10, dtype=np.int64)
    result = fit_poisson_single_change_point(
        counts,
        min_segment_size=2,
    )

    assert result.null_rate == pytest.approx(0.0)
    assert np.all(np.isfinite(result.profile_log_likelihood))
    assert result.likelihood_ratio == pytest.approx(0.0)


@pytest.mark.parametrize("min_segment_size", [0, -1])
def test_rejects_invalid_minimum_segment_range(
    min_segment_size: int,
) -> None:
    with pytest.raises(ValueError):
        fit_poisson_single_change_point(
            np.arange(10, dtype=np.int64),
            min_segment_size=min_segment_size,
        )


def test_rejects_minimum_segment_with_no_candidate() -> None:
    with pytest.raises(ValueError, match="no admissible"):
        fit_poisson_single_change_point(
            np.arange(10, dtype=np.int64),
            min_segment_size=6,
        )


def test_rejects_non_integer_minimum_segment() -> None:
    with pytest.raises(TypeError):
        fit_poisson_single_change_point(
            np.arange(10, dtype=np.int64),
            min_segment_size=2.5,  # type: ignore[arg-type]
        )
