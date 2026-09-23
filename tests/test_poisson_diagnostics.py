"""Tests for Poisson model diagnostics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import DesignMatrix, fit_poisson_loglinear, poisson_diagnostics


def _fit_intercept_only(counts: list[int]) -> object:
    design = DesignMatrix(
        values=np.ones((len(counts), 1)),
        columns=("intercept",),
    )
    return fit_poisson_loglinear(design, counts)


def test_diagnostics_match_direct_pearson_calculation() -> None:
    result = _fit_intercept_only([1, 2, 3, 4, 5, 6])
    diagnostics = poisson_diagnostics(cast("PoissonRegressionResult", result))

    expected_residuals = (
        result.observed_counts.astype(float) - result.fitted_intensity
    ) / np.sqrt(result.fitted_intensity)
    expected_chi_square = float(expected_residuals @ expected_residuals)

    np.testing.assert_allclose(diagnostics.pearson_residuals, expected_residuals)
    assert diagnostics.pearson_chi_square == pytest.approx(expected_chi_square)
    assert diagnostics.residual_degrees_of_freedom == result.nobs - result.nparams
    assert diagnostics.pearson_dispersion == pytest.approx(
        expected_chi_square / diagnostics.residual_degrees_of_freedom
    )


def test_deviance_residuals_reconstruct_model_deviance() -> None:
    result = _fit_intercept_only([0, 1, 2, 4, 8])
    diagnostics = poisson_diagnostics(cast("PoissonRegressionResult", result))

    reconstructed = float(diagnostics.deviance_residuals @ diagnostics.deviance_residuals)
    assert reconstructed == pytest.approx(result.deviance)
    assert diagnostics.deviance_dispersion == pytest.approx(
        result.deviance / diagnostics.residual_degrees_of_freedom
    )


def test_variance_to_mean_ratio_is_raw_count_summary() -> None:
    counts = [1, 1, 1, 9, 9, 9]
    result = _fit_intercept_only(counts)
    diagnostics = poisson_diagnostics(cast("PoissonRegressionResult", result))

    expected = np.var(counts, ddof=1) / np.mean(counts)
    assert diagnostics.variance_to_mean_ratio == pytest.approx(expected)


def test_diagnostic_arrays_are_immutable() -> None:
    result = _fit_intercept_only([1, 2, 3, 4])
    diagnostics = poisson_diagnostics(cast("PoissonRegressionResult", result))

    assert not diagnostics.pearson_residuals.flags.writeable
    assert not diagnostics.deviance_residuals.flags.writeable


def test_observed_counts_are_retained_and_immutable() -> None:
    result = _fit_intercept_only([1, 2, 3, 4])
    assert np.array_equal(result.observed_counts, [1, 2, 3, 4])
    assert not result.observed_counts.flags.writeable


def test_diagnostics_reject_non_result() -> None:
    with pytest.raises(TypeError, match="PoissonRegressionResult"):
        poisson_diagnostics(cast("PoissonRegressionResult", object()))


from count_time_series import PoissonRegressionResult
