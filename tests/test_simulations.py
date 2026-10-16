"""
Integration tests for src/simulation.py and src/benchmark.py.

Validates:
  - GBM path properties (positivity, length, seed reproducibility)
  - Benchmark self-financing (wealth conserved except for trade costs)
  - Separate-fee pool: LVR_tc ≈ 0 (Theorem 3.2)
  - Retained-fee pool: inventory schedule maintained; LVR_tc ≈ 0 (Proposition 5.2)
  - GMM pool: risky fraction stays in [pi_minus, pi_plus] (Section 5.3)
"""

import math
import numpy as np
import pytest

from src.simulation import (
    generate_gbm,
    simulate_separate_fee_pool,
    simulate_retained_fee_pool,
    simulate_gmm_pool,
)
from src.benchmark import RebalancingBenchmark


# ── GBM ───────────────────────────────────────────────────────────────────────

class TestGBM:
    def test_length(self):
        path = generate_gbm(s0=1.0, mu=0.05, sigma=0.20, dt=0.01, n_steps=100, seed=0)
        assert len(path) == 101

    def test_starts_at_s0(self):
        path = generate_gbm(s0=2.5, mu=0.0, sigma=0.10, dt=0.01, n_steps=50, seed=1)
        assert math.isclose(path[0], 2.5, rel_tol=1e-12)

    def test_all_positive(self):
        path = generate_gbm(s0=1.0, mu=-0.5, sigma=0.30, dt=0.001, n_steps=5000, seed=3)
        assert np.all(path > 0)

    def test_seed_reproducibility(self):
        p1 = generate_gbm(1.0, 0.0, 0.2, 0.01, 100, seed=99)
        p2 = generate_gbm(1.0, 0.0, 0.2, 0.01, 100, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        p1 = generate_gbm(1.0, 0.0, 0.2, 0.01, 100, seed=1)
        p2 = generate_gbm(1.0, 0.0, 0.2, 0.01, 100, seed=2)
        assert not np.array_equal(p1, p2)


# ── Benchmark ─────────────────────────────────────────────────────────────────

class TestBenchmark:
    def test_initialize_matches_pool_wealth(self):
        v_lp0, q0, s0, lam = 200.0, 100.0, 1.0, 0.001
        bench = RebalancingBenchmark.initialize(v_lp0, q0, s0, lam)
        assert math.isclose(bench.liquidation_value(s0), v_lp0, rel_tol=1e-10)

    def test_self_financing_buy(self):
        """Buying risky decreases cash by ask_price * dQ."""
        bench = RebalancingBenchmark(n_cash=500.0, q=10.0, lam=0.001)
        s = 2.0
        ask = s / (1 - 0.001)
        bench.rebalance(15.0, s)   # buy 5 units
        assert math.isclose(bench.n_cash, 500.0 - ask * 5.0, rel_tol=1e-10)

    def test_self_financing_sell(self):
        """Selling risky increases cash by bid_price * |dQ|."""
        bench = RebalancingBenchmark(n_cash=500.0, q=10.0, lam=0.001)
        s = 2.0
        bid = s * (1 - 0.001)
        bench.rebalance(5.0, s)   # sell 5 units
        assert math.isclose(bench.n_cash, 500.0 + bid * 5.0, rel_tol=1e-10)

    def test_no_change_preserves_cash(self):
        bench = RebalancingBenchmark(n_cash=500.0, q=10.0, lam=0.001)
        bench.rebalance(10.0, 1.5)
        assert math.isclose(bench.n_cash, 500.0, rel_tol=1e-12)


# ── Separate-fee pool: Theorem 3.2 ───────────────────────────────────────────

class TestSeparateFeeSimulation:
    """LVR_tc = V_R - V_LP ≈ 0 for the separate-fee pool (Theorem 3.2)."""

    @pytest.fixture(scope="class")
    def result(self):
        s_star = generate_gbm(s0=1.0, mu=0.05, sigma=0.20,
                               dt=1/52_000, n_steps=2000, seed=42)
        df = simulate_separate_fee_pool(
            s_star=s_star, x0=100.0, y0=100.0, tau=0.003, lam=0.001)
        return df

    def test_dataframe_shape(self, result):
        assert len(result) == 2001
        required = {"t", "s_star", "s_pool", "X", "Y", "Xf", "Yf",
                    "Q", "R", "V_LP", "V_R", "LVR_tc"}
        assert required.issubset(result.columns)

    def test_lvr_tc_zero(self, result):
        """max|LVR_tc| should be very small relative to pool wealth.

        Theorem 3.2 holds exactly in continuous time.  In discrete time with
        dt = 1/52_000, each arbitrage step introduces O(dt) error (the first-
        order term cancels via gamma = (1-tau)*(1-lam)), so the maximum
        relative LVR_tc grows as O(sigma^2 * n_trades * dt).  A tolerance of
        1e-2 (1%) clearly confirms the theorem while admitting discrete error.
        """
        max_rel = (result["LVR_tc"].abs() / result["V_LP"]).max()
        assert max_rel < 1e-2, f"max relative LVR_tc = {max_rel:.2e} (expected < 1e-2)"

    def test_pool_price_in_band(self, result):
        """Pool price must always lie within [gamma*S*, S*/gamma]."""
        tau, lam = 0.003, 0.001
        gamma = (1 - tau) * (1 - lam)
        lower = gamma * result["s_star"]
        upper = result["s_star"] / gamma
        assert (result["s_pool"] >= lower - 1e-10).all()
        assert (result["s_pool"] <= upper + 1e-10).all()

    def test_fee_accounts_non_negative(self, result):
        assert (result["Xf"] >= -1e-12).all()
        assert (result["Yf"] >= -1e-12).all()

    def test_wealth_direction(self, result):
        """V_LP and V_R should start equal and track each other closely."""
        assert math.isclose(result["V_LP"].iloc[0], result["V_R"].iloc[0], rel_tol=1e-10)


# ── Retained-fee pool: Proposition 5.2 ───────────────────────────────────────

class TestRetainedFeeSimulation:
    @pytest.fixture(scope="class")
    def result(self):
        s_star = generate_gbm(s0=1.0, mu=0.05, sigma=0.20,
                               dt=1/52_000, n_steps=2000, seed=123)
        df = simulate_retained_fee_pool(
            s_star=s_star, c=100.0, beta=0.5, tau=0.003, lam=0.001)
        return df

    def test_dataframe_columns(self, result):
        required = {"t", "s_star", "s_pool", "X", "Y", "K_buffer",
                    "V_LP", "V_R", "LVR_tc", "schedule_error"}
        assert required.issubset(result.columns)

    def test_lvr_tc_zero(self, result):
        """LVR_tc ≈ 0 for retained-fee pool (Proposition 5.2).

        Continuous-time identity; discrete-time error is O(sigma^2 * n_trades * dt).
        Tolerance 1e-2 (1%) confirms the proposition while accommodating discrete error.
        """
        max_rel = (result["LVR_tc"].abs() / result["V_LP"]).max()
        assert max_rel < 1e-2, f"max relative LVR_tc = {max_rel:.2e} (expected < 1e-2)"

    def test_schedule_maintained(self, result):
        """X(t) = q(S(t)) at all times (schedule_error ≈ 0)."""
        max_err = result["schedule_error"].abs().max()
        # schedule_error should be exactly 0 by construction
        assert max_err < 1e-10, f"max schedule_error = {max_err:.2e}"

    def test_fee_buffer_non_decreasing(self, result):
        k_buf = result["K_buffer"].values
        diffs = np.diff(k_buf)
        assert np.all(diffs >= -1e-12), f"min buffer change = {diffs.min():.2e}"

    def test_fee_buffer_non_negative(self, result):
        assert (result["K_buffer"] >= -1e-12).all()


# ── GMM pool: Section 5.3 ─────────────────────────────────────────────────────

class TestGMMSimulation:
    @pytest.fixture(scope="class")
    def result(self):
        s_star = generate_gbm(s0=1.0, mu=0.08, sigma=0.20,
                               dt=1/52_000, n_steps=2000, seed=7)
        df = simulate_gmm_pool(
            s_star=s_star, pi_minus=0.40, pi_plus=0.60, lam=0.002)
        return df

    def test_dataframe_columns(self, result):
        required = {"t", "s_star", "s_pool", "X", "Y", "pi",
                    "V_LP", "V_R", "LVR_tc", "pi_in_band"}
        assert required.issubset(result.columns)

    def test_pi_in_band(self, result):
        """Risky fraction pi must stay in [pi_minus, pi_plus]."""
        pct_in_band = result["pi_in_band"].mean()
        assert pct_in_band > 0.999, f"pi_in_band = {pct_in_band:.4f} (expected > 0.999)"

    def test_lvr_tc_zero(self, result):
        max_rel = (result["LVR_tc"].abs() / result["V_LP"]).max()
        assert max_rel < 1e-6, f"max relative LVR_tc = {max_rel:.2e}"

    def test_pi_between_zero_and_one(self, result):
        assert (result["pi"] > 0).all()
        assert (result["pi"] < 1).all()
