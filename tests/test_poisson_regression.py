"""Tests for Poisson log-linear regression."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import ArrayLike

from poisson_time_series import DesignMatrix, fit_poisson_loglinear


def _intercept_design(nobs: int) -> DesignMatrix:
    return DesignMatrix(np.ones((nobs, 1)), ("intercept",))


def test_intercept_only_model_recovers_sample_mean() -> None:
    counts = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    result = fit_poisson_loglinear(_intercept_design(counts.size), counts)

    assert result.converged
    assert result.coefficients[0] == pytest.approx(np.log(np.mean(counts)), abs=1e-10)
    np.testing.assert_allclose(result.fitted_intensity, np.mean(counts), atol=1e-10)
    assert result.nparams == 1
    assert result.nobs == counts.size
    assert result.deviance >= 0.0
    assert np.isfinite(result.log_likelihood)


def test_exposure_changes_intercept_interpretation() -> None:
    counts = np.array([2, 4, 6, 8], dtype=np.int64)
    exposure = np.array([1.0, 2.0, 3.0, 4.0])
    result = fit_poisson_loglinear(
        _intercept_design(counts.size),
        counts,
        exposure=exposure,
    )

    expected_rate = counts.sum() / exposure.sum()
    assert result.coefficients[0] == pytest.approx(np.log(expected_rate), abs=1e-10)
    np.testing.assert_allclose(result.fitted_intensity, exposure * expected_rate, atol=1e-10)


def test_two_parameter_model_recovers_simulated_signal() -> None:
    x = np.linspace(-1.0, 1.0, 200)
    design = DesignMatrix(
        np.column_stack([np.ones_like(x), x]),
        ("intercept", "trend"),
    )
    beta = np.array([1.0, 0.7])
    mean = np.exp(design.values @ beta)
    counts = np.random.default_rng(42).poisson(mean)

    result = fit_poisson_loglinear(design, counts)

    assert result.converged
    np.testing.assert_allclose(result.coefficients, beta, atol=0.15)
    assert np.all(result.standard_errors > 0.0)
    assert not result.coefficients.flags.writeable
    assert not result.covariance.flags.writeable


def test_predict_mean_requires_matching_columns() -> None:
    result = fit_poisson_loglinear(_intercept_design(4), [1, 2, 3, 4])
    future = DesignMatrix(np.ones((2, 1)), ("intercept",))
    predictions = result.predict_mean(future, exposure=[1.0, 2.0])

    assert predictions.shape == (2,)
    assert predictions[1] == pytest.approx(2.0 * predictions[0])
    assert not predictions.flags.writeable


def test_predict_mean_rejects_mismatched_columns() -> None:
    result = fit_poisson_loglinear(_intercept_design(4), [1, 2, 3, 4])
    wrong = DesignMatrix(np.ones((1, 1)), ("wrong",))
    with pytest.raises(ValueError, match="columns"):
        result.predict_mean(wrong)


def test_predict_mean_rejects_non_design() -> None:
    result = fit_poisson_loglinear(_intercept_design(4), [1, 2, 3, 4])
    with pytest.raises(TypeError, match="DesignMatrix"):
        result.predict_mean(cast(DesignMatrix, object()))


@pytest.mark.parametrize(
    "counts",
    [[], [1.0, np.nan], [-1, 2], [1.5, 2.0], [[1, 2], [3, 4]]],
)
def test_fit_rejects_invalid_counts(counts: ArrayLike) -> None:
    with pytest.raises(ValueError):
        fit_poisson_loglinear(_intercept_design(2), counts)


def test_fit_rejects_non_numeric_counts() -> None:
    with pytest.raises(TypeError, match="numeric"):
        fit_poisson_loglinear(_intercept_design(2), ["a", "b"])


def test_fit_rejects_count_length_mismatch() -> None:
    with pytest.raises(ValueError, match="counts length"):
        fit_poisson_loglinear(_intercept_design(3), [1, 2])


@pytest.mark.parametrize(
    "exposure",
    [[1.0], [1.0, 0.0], [1.0, -1.0], [1.0, np.nan], [[1.0, 2.0]]],
)
def test_fit_rejects_invalid_exposure(exposure: ArrayLike) -> None:
    with pytest.raises(ValueError):
        fit_poisson_loglinear(_intercept_design(2), [1, 2], exposure=exposure)


def test_fit_rejects_non_numeric_exposure() -> None:
    with pytest.raises(TypeError, match="numeric"):
        fit_poisson_loglinear(_intercept_design(2), [1, 2], exposure=["a", "b"])


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.nan])
def test_fit_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError):
        fit_poisson_loglinear(_intercept_design(2), [1, 2], tolerance=tolerance)


@pytest.mark.parametrize("max_iterations", [0, -1])
def test_fit_rejects_invalid_iteration_range(max_iterations: int) -> None:
    with pytest.raises(ValueError):
        fit_poisson_loglinear(
            _intercept_design(2),
            [1, 2],
            max_iterations=max_iterations,
        )


def test_fit_rejects_non_integer_iterations() -> None:
    with pytest.raises(TypeError):
        fit_poisson_loglinear(
            _intercept_design(2),
            [1, 2],
            max_iterations=cast(int, 1.5),
        )


def test_fit_can_report_non_convergence_without_hiding_result() -> None:
    result = fit_poisson_loglinear(
        _intercept_design(4),
        [1, 2, 3, 4],
        max_iterations=1,
        tolerance=1e-20,
    )
    assert result.iterations == 1
    assert result.converged is False


@pytest.mark.parametrize(
    ("values", "columns"),
    [
        (np.ones(3), ("x",)),
        (np.empty((0, 1)), ("x",)),
        (np.array([[1.0], [np.nan]]), ("x",)),
        (np.ones((2, 2)), ("x",)),
        (np.ones((2, 2)), ("x", "x")),
        (np.ones((3, 2)), ("a", "b")),
    ],
)
def test_design_matrix_rejects_invalid_inputs(
    values: np.ndarray,
    columns: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        DesignMatrix(values, columns)
