"""Tests for probabilistic count forecast evaluation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from scipy.stats import nbinom, poisson  # type: ignore[import-untyped]

from count_time_series import (
    negative_binomial_log_score,
    pit_calibration,
    poisson_log_score,
    randomized_pit_negative_binomial,
    randomized_pit_poisson,
)


def test_poisson_log_score_matches_scipy_logpmf() -> None:
    counts = np.array([0, 1, 3, 5])
    mean = np.array([0.5, 1.5, 2.5, 5.0])

    scores = poisson_log_score(counts, mean)

    np.testing.assert_allclose(scores, -poisson.logpmf(counts, mean))
    assert not scores.flags.writeable


def test_negative_binomial_log_score_matches_scipy_parameterization() -> None:
    counts = np.array([0, 2, 4, 7])
    mean = np.array([1.0, 2.0, 3.0, 6.0])
    dispersion = 0.4
    size = 1.0 / dispersion
    probability = size / (size + mean)

    scores = negative_binomial_log_score(
        counts,
        mean,
        dispersion=dispersion,
    )

    np.testing.assert_allclose(
        scores,
        -nbinom.logpmf(counts, size, probability),
    )


def test_randomized_poisson_pit_is_reproducible_with_generator() -> None:
    counts = [0, 1, 2, 3]
    mean = [0.5, 1.0, 2.0, 3.0]

    first = randomized_pit_poisson(
        counts,
        mean,
        rng=np.random.default_rng(42),
    )
    second = randomized_pit_poisson(
        counts,
        mean,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_allclose(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))
    assert not first.flags.writeable


def test_randomized_nb2_pit_is_reproducible_with_generator() -> None:
    counts = [0, 1, 2, 5]
    mean = [0.8, 1.2, 2.5, 4.5]

    first = randomized_pit_negative_binomial(
        counts,
        mean,
        dispersion=0.3,
        rng=np.random.default_rng(7),
    )
    second = randomized_pit_negative_binomial(
        counts,
        mean,
        dispersion=0.3,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_randomized_pit_lies_inside_discrete_cdf_jump() -> None:
    counts = np.array([0, 1, 3])
    mean = np.array([1.0, 2.0, 4.0])
    pit = randomized_pit_poisson(
        counts,
        mean,
        rng=np.random.default_rng(3),
    )

    lower = poisson.cdf(counts - 1, mean)
    upper = poisson.cdf(counts, mean)

    assert np.all(pit >= lower)
    assert np.all(pit <= upper)


def test_pit_calibration_matches_histogram_and_uniform_reference() -> None:
    values = np.linspace(0.005, 0.995, 100)
    calibration = pit_calibration(values, bins=10)

    np.testing.assert_allclose(calibration.bin_frequencies, np.full(10, 0.1))
    assert calibration.mean == pytest.approx(0.5)
    assert calibration.variance == pytest.approx(np.var(values, ddof=1))
    assert calibration.ks_statistic >= 0.0
    assert 0.0 <= calibration.ks_p_value <= 1.0
    assert calibration.nobs == 100
    assert calibration.bins == 10
    assert not calibration.values.flags.writeable
    assert not calibration.bin_edges.flags.writeable
    assert not calibration.bin_frequencies.flags.writeable


@pytest.mark.parametrize(
    "mean",
    [[1.0], [1.0, 0.0], [1.0, -1.0], [1.0, np.nan], [[1.0, 2.0]]],
)
def test_poisson_log_score_rejects_invalid_mean(mean: object) -> None:
    with pytest.raises(ValueError):
        poisson_log_score([1, 2], cast(object, mean))


def test_poisson_log_score_rejects_non_numeric_mean() -> None:
    with pytest.raises(TypeError):
        poisson_log_score([1, 2], ["a", "b"])


@pytest.mark.parametrize("dispersion", [0.0, -1.0, np.nan, np.inf])
def test_nb2_score_rejects_invalid_dispersion(dispersion: float) -> None:
    with pytest.raises(ValueError):
        negative_binomial_log_score(
            [1, 2],
            [1.0, 2.0],
            dispersion=dispersion,
        )


def test_randomized_pit_rejects_invalid_rng() -> None:
    with pytest.raises(TypeError):
        randomized_pit_poisson(
            [1, 2],
            [1.0, 2.0],
            rng=cast(np.random.Generator, object()),
        )


@pytest.mark.parametrize(
    "values",
    [
        [0.5],
        [0.1, np.nan],
        [-0.1, 0.5],
        [0.5, 1.1],
        [[0.1, 0.2], [0.3, 0.4]],
    ],
)
def test_pit_calibration_rejects_invalid_values(values: object) -> None:
    with pytest.raises(ValueError):
        pit_calibration(cast(object, values))


def test_pit_calibration_rejects_non_numeric_values() -> None:
    with pytest.raises(TypeError):
        pit_calibration(["a", "b"])


@pytest.mark.parametrize("bins", [0, 1, -1])
def test_pit_calibration_rejects_invalid_bin_range(bins: int) -> None:
    with pytest.raises(ValueError):
        pit_calibration([0.2, 0.8], bins=bins)


def test_pit_calibration_rejects_non_integer_bins() -> None:
    with pytest.raises(TypeError):
        pit_calibration([0.2, 0.8], bins=cast(int, 2.5))
