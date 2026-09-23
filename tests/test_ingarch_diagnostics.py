"""Tests for temporal INGARCH residual diagnostics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from count_time_series import (
    PoissonINGARCHResult,
    fit_poisson_ingarch,
    ingarch_diagnostics,
    simulate_poisson_ingarch,
)


def _fit() -> PoissonINGARCHResult:
    simulation = simulate_poisson_ingarch(
        1200,
        omega=1.0,
        alpha=0.20,
        beta=0.55,
        burn_in=500,
        rng=np.random.default_rng(42),
    )
    return fit_poisson_ingarch(simulation.counts)


def test_ingarch_diagnostics_exclude_fixed_initial_observation() -> None:
    result = _fit()
    diagnostics = ingarch_diagnostics(result, lags=10)

    assert diagnostics.nobs == result.nobs - 1
    expected = (
        result.observed_counts[1:] - result.fitted_intensity[1:]
    ) / np.sqrt(result.fitted_intensity[1:])
    np.testing.assert_allclose(diagnostics.pearson_residuals, expected)
    assert not diagnostics.pearson_residuals.flags.writeable
    assert not diagnostics.autocorrelation.flags.writeable


def test_ingarch_acf_has_unit_lag_zero() -> None:
    diagnostics = ingarch_diagnostics(_fit(), lags=8)

    assert diagnostics.autocorrelation.shape == (9,)
    assert diagnostics.autocorrelation[0] == pytest.approx(1.0)


def test_ingarch_ljung_box_matches_direct_formula() -> None:
    diagnostics = ingarch_diagnostics(_fit(), lags=6, model_df=3)
    nobs = diagnostics.nobs
    expected = nobs * (nobs + 2) * sum(
        diagnostics.autocorrelation[lag] ** 2 / (nobs - lag)
        for lag in range(1, 7)
    )

    assert diagnostics.ljung_box_statistic == pytest.approx(expected)
    assert diagnostics.ljung_box_degrees_of_freedom == 3
    assert diagnostics.model_df == 3
    assert 0.0 <= diagnostics.ljung_box_p_value <= 1.0


def test_ingarch_residual_scale_is_reasonable_for_correct_model() -> None:
    diagnostics = ingarch_diagnostics(_fit(), lags=10)

    assert abs(diagnostics.residual_mean) < 0.15
    assert diagnostics.residual_variance == pytest.approx(1.0, abs=0.20)


@pytest.mark.parametrize("lags", [0, -1, 1199])
def test_ingarch_diagnostics_reject_invalid_lags(lags: int) -> None:
    result = _fit()
    with pytest.raises(ValueError):
        ingarch_diagnostics(result, lags=lags)


def test_ingarch_diagnostics_reject_non_integer_lags() -> None:
    with pytest.raises(TypeError):
        ingarch_diagnostics(_fit(), lags=cast(int, 2.5))


@pytest.mark.parametrize("model_df", [-1, 4])
def test_ingarch_diagnostics_reject_invalid_model_df(model_df: int) -> None:
    with pytest.raises(ValueError):
        ingarch_diagnostics(_fit(), lags=4, model_df=model_df)


def test_ingarch_diagnostics_reject_non_integer_model_df() -> None:
    with pytest.raises(TypeError):
        ingarch_diagnostics(
            _fit(),
            lags=4,
            model_df=cast(int, 1.5),
        )


def test_ingarch_diagnostics_reject_non_result() -> None:
    with pytest.raises(TypeError, match="PoissonINGARCHResult"):
        ingarch_diagnostics(
            cast(PoissonINGARCHResult, object()),
            lags=4,
        )
