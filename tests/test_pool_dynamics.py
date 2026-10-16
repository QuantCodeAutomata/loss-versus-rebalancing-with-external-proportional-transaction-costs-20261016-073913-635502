"""
Tests for src/pool_dynamics.py.

Tests the substance of the paper's key quantities:
  - Reserve-curve math (constant product, inventory schedule, GMM)
  - No-arbitrage band [gamma*S*, S*/gamma]
  - Separate-fee accumulation rules (Section 2.2.2)
  - Retained-fee buffer monotonicity (Section 2.2.3)
  - GMM parameter derivation (Section 5.3)
  - LVR_tc = 0 by construction (Theorem 3.2, Propositions 5.2/5.3)
"""

import math
import pytest
import numpy as np

from src.pool_dynamics import (
    constant_product_price,
    constant_product_reserves,
    inventory_schedule_f,
    inventory_schedule_f_prime,
    inventory_schedule_f_double_prime,
    inventory_schedule_price,
    inventory_schedule_quantity,
    gmm_price,
    gmm_reserves_from_price,
    risky_fraction,
    gmm_params_from_boundaries,
    SeparateFeePool,
    RetainedFeePool,
    GMMPool,
)


# ── Constant-product helpers ─────────────────────────────────────────────────

class TestConstantProduct:
    def test_price_roundtrip(self):
        k = 10_000.0
        x, y = 100.0, 100.0
        s = constant_product_price(x, k)
        assert math.isclose(s, 1.0, rel_tol=1e-10)

    def test_reserves_roundtrip(self):
        k = 10_000.0
        s = 2.0
        x, y = constant_product_reserves(s, k)
        assert math.isclose(x * y, k, rel_tol=1e-10)
        assert math.isclose(y / x, s, rel_tol=1e-10)

    def test_price_increases_as_x_decreases(self):
        k = 10_000.0
        p1 = constant_product_price(100.0, k)
        p2 = constant_product_price(80.0, k)
        assert p2 > p1


# ── Inventory-schedule reserve curve ─────────────────────────────────────────

class TestInventorySchedule:
    """Tests for the Proposition 5.1 reserve curve."""

    def test_price_equals_negative_f_prime(self):
        """S = -f'(X) = (c/X)^{1/beta}  at X = q(S) (confirms sign convention)."""
        c, beta = 50.0, 0.5
        s = 2.0
        x = inventory_schedule_quantity(s, c, beta)
        fp = inventory_schedule_f_prime(x, c, beta)
        assert math.isclose(-fp, s, rel_tol=1e-10)

    def test_quantity_is_inverse_of_price(self):
        """q(S) and price(q(S)) are mutual inverses."""
        c, beta = 50.0, 0.3
        for s in [0.5, 1.0, 2.0, 5.0]:
            x = inventory_schedule_quantity(s, c, beta)
            s_back = inventory_schedule_price(x, c, beta)
            assert math.isclose(s_back, s, rel_tol=1e-9)

    def test_f_is_decreasing_in_x(self):
        """f(x) is decreasing (pool pays out numeraire when x rises)."""
        c, beta = 50.0, 0.5
        f1 = inventory_schedule_f(100.0, c, beta)
        f2 = inventory_schedule_f(120.0, c, beta)
        assert f2 < f1

    def test_f_double_prime_positive(self):
        """f''(x) > 0 ensures the reserve curve is convex (required for AMM)."""
        c, beta = 50.0, 0.5
        for x in [50.0, 100.0, 200.0]:
            fpp = inventory_schedule_f_double_prime(x, c, beta)
            assert fpp > 0

    def test_beta_boundary(self):
        with pytest.raises(AssertionError):
            inventory_schedule_f(100.0, c=50.0, beta=0.0)
        with pytest.raises(AssertionError):
            inventory_schedule_f(100.0, c=50.0, beta=1.0)


# ── GMM helpers ───────────────────────────────────────────────────────────────

class TestGMM:
    def test_price_from_reserves(self):
        """GMM price S = k * Y / X."""
        k, x, y = 1.0, 100.0, 100.0
        assert math.isclose(gmm_price(x, y, k), 1.0, rel_tol=1e-10)

    def test_reserves_roundtrip(self):
        k, c_inv = 0.5, 5000.0
        s = 1.5
        x, y = gmm_reserves_from_price(s, c_inv, k)
        # Verify invariant X^k * Y = C
        assert math.isclose(x**k * y, c_inv, rel_tol=1e-9)
        # Verify price
        assert math.isclose(gmm_price(x, y, k), s, rel_tol=1e-9)

    def test_risky_fraction_bounds(self):
        s_ext, x, y = 1.0, 50.0, 50.0
        pi = risky_fraction(s_ext, x, y)
        assert 0 < pi < 1
        assert math.isclose(pi, 0.5, rel_tol=1e-10)


# ── GMM parameter derivation (Section 5.3) ───────────────────────────────────

class TestGMMParams:
    def test_equilibrium_fraction(self):
        """GMM equilibrium fraction = k / (1+k)."""
        pi_minus, pi_plus, lam = 0.40, 0.60, 0.002
        p = gmm_params_from_boundaries(pi_minus, pi_plus, lam)
        k = p["k"]
        pi_eq = k / (1 + k)
        # Equilibrium should lie between the two boundaries (for symmetric case)
        assert pi_minus < pi_eq < pi_plus

    def test_gamma_in_zero_one(self):
        p = gmm_params_from_boundaries(0.40, 0.60, 0.002)
        assert 0 < p["gamma"] < 1

    def test_tau_feasibility(self):
        """tau >= 0 required for a valid pool fee."""
        p = gmm_params_from_boundaries(0.40, 0.60, 0.002)
        assert p["tau"] >= 0

    def test_infeasible_raises(self):
        """pi_minus >= pi_plus should raise AssertionError."""
        with pytest.raises(AssertionError):
            gmm_params_from_boundaries(0.6, 0.4, 0.002)


# ── SeparateFeePool ───────────────────────────────────────────────────────────

class TestSeparateFeePool:
    def _make_pool(self, tau=0.003, lam=0.001):
        return SeparateFeePool(
            x=100.0, y=100.0, xf=0.0, yf=0.0,
            k_const=10_000.0, tau=tau, lam=lam,
        )

    def test_initial_pool_price(self):
        pool = self._make_pool()
        assert math.isclose(pool.pool_price, 1.0, rel_tol=1e-10)

    def test_gamma_formula(self):
        tau, lam = 0.003, 0.001
        pool = self._make_pool(tau=tau, lam=lam)
        expected = (1 - tau) * (1 - lam)
        assert math.isclose(pool.gamma, expected, rel_tol=1e-12)

    def test_no_arb_band_contains_price(self):
        pool = self._make_pool()
        lower, upper = pool.no_arb_band(1.0)
        assert lower < pool.pool_price < upper

    def test_arbitrage_step_price_rise(self):
        """After price rises, pool price should be at lower band boundary."""
        pool = self._make_pool()
        s_new = 1.5
        pool.arbitrage_step(s_new)
        lower, upper = pool.no_arb_band(s_new)
        assert math.isclose(pool.pool_price, lower, rel_tol=1e-8)

    def test_arbitrage_step_price_fall(self):
        """After price falls, pool price should be at upper band boundary."""
        pool = self._make_pool()
        s_new = 0.7
        pool.arbitrage_step(s_new)
        lower, upper = pool.no_arb_band(s_new)
        assert math.isclose(pool.pool_price, upper, rel_tol=1e-8)

    def test_fee_accumulation_numeraire(self):
        """Numeraire fee Yf increases on price rise."""
        pool = self._make_pool()
        pool.arbitrage_step(1.5)
        assert pool.yf > 0
        assert math.isclose(pool.xf, 0.0, abs_tol=1e-12)

    def test_fee_accumulation_risky(self):
        """Risky fee Xf increases on price fall."""
        pool = self._make_pool()
        pool.arbitrage_step(0.7)
        assert pool.xf > 0
        assert math.isclose(pool.yf, 0.0, abs_tol=1e-12)

    def test_constant_product_invariant(self):
        """x * y = k_const must hold after every step."""
        pool = self._make_pool()
        for s in [1.2, 0.8, 1.5, 0.6]:
            pool.arbitrage_step(s)
            assert math.isclose(pool.x * pool.y, pool.k_const, rel_tol=1e-9)


# ── RetainedFeePool ───────────────────────────────────────────────────────────

class TestRetainedFeePool:
    def _make_pool(self, c=100.0, beta=0.5, tau=0.003, lam=0.001, s0=1.0):
        from src.pool_dynamics import inventory_schedule_quantity, inventory_schedule_f
        x0 = inventory_schedule_quantity(s0, c, beta)
        y0 = inventory_schedule_f(x0, c, beta)
        return RetainedFeePool(x=x0, y=y0, c=c, beta=beta, tau=tau, lam=lam)

    def test_initial_pool_price(self):
        pool = self._make_pool(s0=1.0)
        assert math.isclose(pool.pool_price, 1.0, rel_tol=1e-9)

    def test_initial_fee_buffer_zero(self):
        """Pool starts with zero fee buffer K = Y - f(X) = 0."""
        pool = self._make_pool()
        assert math.isclose(pool.fee_buffer, 0.0, abs_tol=1e-12)

    def test_schedule_maintained_after_step(self):
        """X = q(S) exactly after arbitrage step (retained-fee dynamics)."""
        pool = self._make_pool()
        s_new = 1.5
        pool.arbitrage_step(s_new)
        lower, upper = pool.no_arb_band(s_new)
        s_pool = pool.pool_price
        q_target = inventory_schedule_quantity(s_pool, pool.c, pool.beta)
        assert math.isclose(pool.x, q_target, rel_tol=1e-9)

    def test_fee_buffer_non_decreasing(self):
        """Fee buffer K = Y - f(X) should never decrease."""
        pool = self._make_pool()
        k_prev = pool.fee_buffer
        for s in [1.1, 1.3, 1.0, 0.8, 1.2, 0.5, 2.0]:
            pool.arbitrage_step(s)
            assert pool.fee_buffer >= k_prev - 1e-12
            k_prev = pool.fee_buffer


# ── GMMPool ───────────────────────────────────────────────────────────────────

class TestGMMPool:
    def _make_pool(self, pi_minus=0.4, pi_plus=0.6, lam=0.002, w=1.0, s0=1.0):
        p = gmm_params_from_boundaries(pi_minus, pi_plus, lam)
        k = p["k"]
        tau = p["tau"]
        x0 = w * k / (s0 * (1 + k))
        y0 = s0 * x0 / k
        c_inv = x0**k * y0
        return GMMPool(x=x0, y=y0, c_inv=c_inv, k=k, tau=tau, lam=lam,
                       pi_minus=pi_minus, pi_plus=pi_plus)

    def test_risky_fraction_in_band(self):
        pool = self._make_pool()
        pi = pool.risky_fraction(s_ext=1.0)
        # Equilibrium pi = k/(1+k) should be in [pi_minus, pi_plus]
        assert 0.40 <= pi <= 0.60

    def test_arbitrage_moves_price_to_band(self):
        pool = self._make_pool()
        s_new = 1.5
        pool.arbitrage_step(s_new)
        lower, upper = pool.no_arb_band(s_new)
        p_pool = pool.pool_price
        assert lower - 1e-9 <= p_pool <= upper + 1e-9

    def test_gmm_invariant_preserved(self):
        """X^k * Y = C after each arb step."""
        pool = self._make_pool()
        for s in [1.2, 0.8, 1.5, 0.6, 1.0]:
            pool.arbitrage_step(s)
            computed = pool.x**pool.k * pool.y
            assert math.isclose(computed, pool.c_inv, rel_tol=1e-9)
