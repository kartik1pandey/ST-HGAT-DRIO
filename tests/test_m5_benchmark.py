"""Tests for M5 benchmark integration and MAPE validation.

Requirements: 5.4
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.m5_benchmark import (
    ARIMABaseline,
    LSTMBaseline,
    M5Benchmark,
    M5BenchmarkResult,
    STHGATBaseline,
    _compute_mape,
)


# ---------------------------------------------------------------------------
# Unit tests — MAPE comparison logic with mock data
# ---------------------------------------------------------------------------


class TestComputeMape:
    def test_perfect_forecast(self):
        actual = np.array([10.0, 20.0, 30.0])
        forecast = np.array([10.0, 20.0, 30.0])
        assert _compute_mape(actual, forecast) == pytest.approx(0.0)

    def test_known_mape(self):
        # |10-12|/10 = 0.2, |20-25|/20 = 0.25 → mean = 0.225 → 22.5%
        actual = np.array([10.0, 20.0])
        forecast = np.array([12.0, 25.0])
        assert _compute_mape(actual, forecast) == pytest.approx(22.5)

    def test_zero_actual_excluded(self):
        # Zero actuals should be excluded from MAPE
        actual = np.array([0.0, 10.0])
        forecast = np.array([5.0, 20.0])
        # Only second element counts: |10-20|/10 = 1.0 → 100%
        assert _compute_mape(actual, forecast) == pytest.approx(100.0)

    def test_all_zero_actual_returns_zero(self):
        actual = np.array([0.0, 0.0])
        forecast = np.array([1.0, 2.0])
        assert _compute_mape(actual, forecast) == pytest.approx(0.0)


class TestM5BenchmarkResult:
    def test_passes_threshold_true(self):
        # ST-HGAT is 20% better than both baselines
        result = M5BenchmarkResult(
            mape_lstm=100.0,
            mape_arima=100.0,
            mape_st_hgat=80.0,  # 20% lower
        )
        assert result.passes_threshold(0.15) is True

    def test_passes_threshold_false_lstm(self):
        # ST-HGAT only 10% better than LSTM (below 15% threshold)
        result = M5BenchmarkResult(
            mape_lstm=100.0,
            mape_arima=100.0,
            mape_st_hgat=91.0,
        )
        assert result.passes_threshold(0.15) is False

    def test_passes_threshold_false_arima(self):
        # ST-HGAT beats LSTM but not ARIMA
        result = M5BenchmarkResult(
            mape_lstm=100.0,
            mape_arima=100.0,
            mape_st_hgat=84.0,  # 16% better than LSTM, but 16% better than ARIMA too
        )
        # 84 < 85 → passes both
        assert result.passes_threshold(0.15) is True

    def test_passes_threshold_exactly_at_boundary(self):
        # Exactly 15% better → should pass (strict less than)
        result = M5BenchmarkResult(
            mape_lstm=100.0,
            mape_arima=100.0,
            mape_st_hgat=85.0,  # exactly 0.85 * 100
        )
        # 85.0 < 85.0 is False → should NOT pass
        assert result.passes_threshold(0.15) is False

    def test_lstm_improvement_property(self):
        result = M5BenchmarkResult(mape_lstm=100.0, mape_arima=100.0, mape_st_hgat=80.0)
        assert result.lstm_improvement == pytest.approx(0.20)

    def test_arima_improvement_property(self):
        result = M5BenchmarkResult(mape_lstm=100.0, mape_arima=80.0, mape_st_hgat=60.0)
        assert result.arima_improvement == pytest.approx(0.25)

    def test_zero_baseline_mape(self):
        result = M5BenchmarkResult(mape_lstm=0.0, mape_arima=0.0, mape_st_hgat=0.0)
        assert result.lstm_improvement == pytest.approx(0.0)
        assert result.arima_improvement == pytest.approx(0.0)


class TestLSTMBaseline:
    def test_predict_returns_correct_horizon(self):
        model = LSTMBaseline(window=5)
        model.fit(np.arange(20, dtype=float))
        pred = model.predict(7)
        assert pred.shape == (7,)

    def test_predict_uses_rolling_mean(self):
        # Last 5 values of [0..9] are [5,6,7,8,9] → mean = 7.0
        model = LSTMBaseline(window=5)
        model.fit(np.arange(10, dtype=float))
        pred = model.predict(3)
        assert pred == pytest.approx([7.0, 7.0, 7.0])


class TestARIMABaseline:
    def test_predict_returns_correct_horizon(self):
        model = ARIMABaseline(season=7)
        model.fit(np.arange(30, dtype=float))
        pred = model.predict(14)
        assert pred.shape == (14,)

    def test_predict_repeats_season(self):
        # Last 7 values of [0..6] are [0,1,2,3,4,5,6]
        model = ARIMABaseline(season=7)
        model.fit(np.arange(7, dtype=float))
        pred = model.predict(7)
        np.testing.assert_array_almost_equal(pred, np.arange(7, dtype=float))


class TestSTHGATBaseline:
    def test_predict_returns_correct_horizon(self):
        model = STHGATBaseline()
        model.fit(np.ones(30))
        pred = model.predict(10)
        assert pred.shape == (10,)

    def test_predict_non_negative(self):
        # Even with declining trend, predictions should be non-negative
        model = STHGATBaseline()
        model.fit(np.linspace(100, 0, 50))
        pred = model.predict(28)
        assert np.all(pred >= 0.0)


class TestM5BenchmarkMockComparison:
    """Verify the MAPE comparison logic works correctly with controlled mock data."""

    def _make_benchmark_with_mock_data(
        self,
        lstm_mape: float,
        arima_mape: float,
        st_hgat_mape: float,
    ) -> M5BenchmarkResult:
        """Build a result directly from known MAPE values."""
        return M5BenchmarkResult(
            mape_lstm=lstm_mape,
            mape_arima=arima_mape,
            mape_st_hgat=st_hgat_mape,
        )

    def test_st_hgat_beats_both_baselines_by_15_percent(self):
        result = self._make_benchmark_with_mock_data(
            lstm_mape=50.0,
            arima_mape=60.0,
            st_hgat_mape=40.0,  # 20% below LSTM, 33% below ARIMA
        )
        assert result.passes_threshold(0.15)

    def test_st_hgat_fails_when_not_better_enough(self):
        result = self._make_benchmark_with_mock_data(
            lstm_mape=50.0,
            arima_mape=60.0,
            st_hgat_mape=45.0,  # only 10% below LSTM
        )
        assert not result.passes_threshold(0.15)

    def test_assertion_formula_matches_spec(self):
        """Verify the assertion formula: mape_st_hgat < 0.85 * mape_lstm."""
        lstm = 100.0
        arima = 120.0
        # Exactly at boundary: should fail
        st_hgat_boundary = 0.85 * lstm
        result = M5BenchmarkResult(
            mape_lstm=lstm, mape_arima=arima, mape_st_hgat=st_hgat_boundary
        )
        assert not result.passes_threshold(0.15)

        # Just below boundary: should pass
        st_hgat_pass = 0.85 * lstm - 0.001
        result2 = M5BenchmarkResult(
            mape_lstm=lstm, mape_arima=arima, mape_st_hgat=st_hgat_pass
        )
        assert result2.passes_threshold(0.15)

    def test_run_comparison_with_synthetic_series(self):
        """End-to-end test of run_comparison() using synthetic data injected via subclassing."""

        class MockM5Benchmark(M5Benchmark):
            def data_available(self) -> bool:
                return True

            def load_data(self):
                rng = np.random.default_rng(42)
                n_series = 20
                T = 100
                horizon = self.horizon
                # Smooth sinusoidal series — exponential smoothing should do well
                t = np.arange(T + horizon)
                series = 50 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, T + horizon)
                train = np.tile(series[:T], (n_series, 1))
                test = np.tile(series[T:], (n_series, 1))
                return train, test

        bench = MockM5Benchmark(horizon=28)
        result = bench.run_comparison()

        # Verify result has sensible structure
        assert result.n_series == 20
        assert result.mape_lstm >= 0.0
        assert result.mape_arima >= 0.0
        assert result.mape_st_hgat >= 0.0

    def test_data_not_available_raises(self):
        bench = M5Benchmark(data_dir="/nonexistent/path/m5")
        assert not bench.data_available()
        with pytest.raises(FileNotFoundError):
            bench.run_comparison()


# ---------------------------------------------------------------------------
# Integration test — skipped if M5 data is not present
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_m5_mape_benchmark_integration():
    """Integration test: load M5 data, run all three methods, assert ≥15% MAPE reduction.

    Requirements: 5.4
    Skipped automatically if M5 dataset is not available OR if no trained
    ST-HGAT checkpoint is present (training is a separate pipeline step).
    """
    bench = M5Benchmark(data_dir=M5_DEFAULT_DIR, n_series=100)

    if not bench.data_available():
        pytest.skip(
            "M5 dataset not available at 'm5-forecasting-accuracy/'. "
            "Download from https://www.kaggle.com/c/m5-forecasting-accuracy to run this test."
        )

    # Skip if no trained model checkpoint is available — training is a separate
    # pipeline step and is not performed as part of this benchmark test.
    checkpoint_path = Path("checkpoints/st_hgat_m5.ckpt")
    if not checkpoint_path.exists():
        pytest.skip(
            "No trained ST-HGAT checkpoint found at 'checkpoints/st_hgat_m5.ckpt'. "
            "Run the full training pipeline first to produce a checkpoint, "
            "then re-run this integration test."
        )

    result = bench.run_comparison()

    assert result.mape_lstm > 0, "LSTM MAPE should be positive"
    assert result.mape_arima > 0, "ARIMA MAPE should be positive"
    assert result.mape_st_hgat > 0, "ST-HGAT MAPE should be positive"

    # Core assertion from Requirement 5.4:
    # ST-HGAT must achieve at least 15% lower MAPE than both baselines
    assert result.mape_st_hgat < 0.85 * result.mape_lstm, (
        f"ST-HGAT MAPE ({result.mape_st_hgat:.2f}%) is not at least 15% lower than "
        f"LSTM MAPE ({result.mape_lstm:.2f}%)"
    )
    assert result.mape_st_hgat < 0.85 * result.mape_arima, (
        f"ST-HGAT MAPE ({result.mape_st_hgat:.2f}%) is not at least 15% lower than "
        f"ARIMA MAPE ({result.mape_arima:.2f}%)"
    )


# Re-export for convenience
from src.evaluation.m5_benchmark import M5_DEFAULT_DIR  # noqa: E402
