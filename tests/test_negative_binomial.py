"""Tests for fixed-dispersion NB2 regression."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import ArrayLike

from count_time_series import (
    DesignMatrix,
    NegativeBinomialRegressionResult,
    fit_negative_binomial_loglinear,
)


def _intercept_design(nobs: int) -> DesignMatrix:
    return DesignMatrix(np.ones((nobs, 1)), ("intercept",))


def test_intercept_only_nb2_recovers_sample_mean() -> None:
    counts = np.array([0, 1, 2, 4, 7, 10], dtype=np.int64)
    result = fit_negative_binomial_loglinear(
        _intercept_design(counts.size),
        counts,
        dispersion=0.5,
    )

    assert result.converged
    expected_mean = float(np.mean(counts))
    assert result.coefficients[0] == pytest.approx(np.log(expected_mean), abs=1e-10)
    np.testing.assert_allclose(result.fitted_mean, expected_mean, atol=1e-10)
    np.testing.assert_allclose(
        result.fitted_variance,
        expected_mean + 0.5 * expected_mean**2,
        atol=1e-10,
    )
    assert result.dispersion == pytest.approx(0.5)
    assert result.deviance >= 0.0
    assert np.isfinite(result.log_likelihood)


def test_nb2_recovers_simulated_regression_signal() -> None:
    rng = np.random.default_rng(42)
    x = np.linspace(-1.0, 1.0, 600)
    design = DesignMatrix(
        np.column_stack([np.ones_like(x), x]),
        ("intercept", "trend"),
    )
    beta = np.array([1.0, 0.6])
    alpha = 0.4
    mean = np.exp(design.values @ beta)
    size = 1.0 / alpha
    probability = size / (size + mean)
    counts = rng.negative_binomial(size, probability)

    result = fit_negative_binomial_loglinear(
        design,
        counts,
        dispersion=alpha,
    )

    assert result.converged
    np.testing.assert_allclose(result.coefficients, beta, atol=0.12)
    assert np.all(result.standard_errors > 0.0)
    assert not result.coefficients.flags.writeable
    assert not result.fitted_variance.flags.writeable


def test_nb2_exposure_scales_predicted_mean() -> None:
    result = fit_negative_binomial_loglinear(
        _intercept_design(4),
        [1, 2, 3, 4],
        dispersion=0.3,
    )
    future = DesignMatrix(np.ones((2, 1)), ("intercept",))

    mean = result.predict_mean(future, exposure=[1.0, 2.0])
    variance = result.predict_variance(future, exposure=[1.0, 2.0])

    assert mean[1] == pytest.approx(2.0 * mean[0])
    np.testing.assert_allclose(variance, mean + 0.3 * mean**2)
    assert not mean.flags.writeable
    assert not variance.flags.writeable


def test_nb2_predict_rejects_mismatched_columns() -> None:
    result = fit_negative_binomial_loglinear(
        _intercept_design(4),
        [1, 2, 3, 4],
        dispersion=0.2,
    )
    wrong = DesignMatrix(np.ones((1, 1)), ("wrong",))

    with pytest.raises(ValueError, match="columns"):
        result.predict_mean(wrong)


def test_nb2_predict_rejects_non_design() -> None:
    result = fit_negative_binomial_loglinear(
        _intercept_design(4),
        [1, 2, 3, 4],
        dispersion=0.2,
    )

    with pytest.raises(TypeError, match="DesignMatrix"):
        result.predict_mean(cast(DesignMatrix, object()))


@pytest.mark.parametrize("dispersion", [0.0, -1.0, np.inf, np.nan])
def test_nb2_rejects_invalid_dispersion_range(dispersion: float) -> None:
    with pytest.raises(ValueError):
        fit_negative_binomial_loglinear(
            _intercept_design(2),
            [1, 2],
            dispersion=dispersion,
        )


def test_nb2_rejects_non_numeric_dispersion() -> None:
    with pytest.raises(TypeError):
        fit_negative_binomial_loglinear(
            _intercept_design(2),
            [1, 2],
            dispersion=cast(float, "bad"),
        )


@pytest.mark.parametrize(
    "counts",
    [[], [1.0, np.nan], [-1, 2], [1.5, 2.0], [[1, 2], [3, 4]]],
)
def test_nb2_rejects_invalid_counts(counts: ArrayLike) -> None:
    with pytest.raises(ValueError):
        fit_negative_binomial_loglinear(
            _intercept_design(2),
            counts,
            dispersion=0.4,
        )


def test_nb2_rejects_count_length_mismatch() -> None:
    with pytest.raises(ValueError, match="counts length"):
        fit_negative_binomial_loglinear(
            _intercept_design(3),
            [1, 2],
            dispersion=0.4,
        )


@pytest.mark.parametrize(
    "exposure",
    [[1.0], [1.0, 0.0], [1.0, -1.0], [1.0, np.nan], [[1.0, 2.0]]],
)
def test_nb2_rejects_invalid_exposure(exposure: ArrayLike) -> None:
    with pytest.raises(ValueError):
        fit_negative_binomial_loglinear(
            _intercept_design(2),
            [1, 2],
            dispersion=0.4,
            exposure=exposure,
        )


def test_nb2_can_report_non_convergence() -> None:
    result = fit_negative_binomial_loglinear(
        _intercept_design(4),
        [1, 2, 3, 4],
        dispersion=0.4,
        max_iterations=1,
        tolerance=1e-20,
    )

    assert result.iterations == 1
    assert result.converged is False


def test_nb2_result_arrays_are_immutable() -> None:
    result: NegativeBinomialRegressionResult = fit_negative_binomial_loglinear(
        _intercept_design(4),
        [1, 2, 3, 4],
        dispersion=0.4,
    )

    assert not result.observed_counts.flags.writeable
    assert not result.covariance.flags.writeable
    assert not result.standard_errors.flags.writeable
