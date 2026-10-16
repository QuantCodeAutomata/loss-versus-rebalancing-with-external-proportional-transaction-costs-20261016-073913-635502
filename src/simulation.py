"""
Simulation utilities: GBM price paths and full pool-simulation orchestrators.

The three orchestrators share one data path and call src/ APIs directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.pool_dynamics import (
    SeparateFeePool,
    RetainedFeePool,
    GMMPool,
    constant_product_reserves,
    inventory_schedule_quantity,
    inventory_schedule_f,
    gmm_reserves_from_price,
    gmm_params_from_boundaries,
)
from src.benchmark import RebalancingBenchmark


# ---------------------------------------------------------------------------
# GBM price-path generator
# ---------------------------------------------------------------------------

def generate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a Geometric Brownian Motion path for the external price S*.

    S*(t + dt) = S*(t) * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z)

    Parameters
    ----------
    s0      : initial price
    mu      : drift
    sigma   : volatility
    dt      : time step
    n_steps : number of steps (path has n_steps + 1 values, including s0)
    seed    : random seed for reproducibility

    Returns
    -------
    array of shape (n_steps + 1,)
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_steps)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_path = np.concatenate([[0.0], np.cumsum(log_returns)])
    return s0 * np.exp(log_path)


# ---------------------------------------------------------------------------
# Simulation 1: Separate-fee pool (Theorem 3.2)
# ---------------------------------------------------------------------------

def simulate_separate_fee_pool(
    s_star: np.ndarray,
    x0: float,
    y0: float,
    tau: float,
    lam: float,
) -> pd.DataFrame:
    """Simulate a separate-fee (Uniswap v3 type) pool under GBM price S*.

    Implements:
      - Constant-product pool X * Y = K with Skorokhod reflection in band
        [gamma*S*, S*/gamma], gamma = (1-tau)*(1-lambda)  (Section 2.1).
      - Separate fee accumulation: dYf = tau/(1-tau) * dY_up  (price rises),
        dXf = tau/(1-tau) * dX_up  (price falls)  (Section 2.2.2).
      - Self-financing benchmark V_R matching Q = X + Xf  (benchmark.py).
      - LVR_tc = V_R_liq - V_LP_liq  (Theorem 3.2: should be ≈ 0).

    Parameters
    ----------
    s_star : external price path, shape (n_steps + 1,)
    x0, y0 : initial risky and numeraire reserves
    tau    : pool fee rate
    lam    : external proportional TC lambda

    Returns
    -------
    DataFrame with columns:
        t, s_star, s_pool, X, Y, Xf, Yf, Q, R,
        V_LP, V_R, LVR_tc
    """
    k_const = x0 * y0
    pool = SeparateFeePool(x=x0, y=y0, xf=0.0, yf=0.0,
                           k_const=k_const, tau=tau, lam=lam)
    v_lp0 = pool.liquidation_value(s_star[0])
    bench = RebalancingBenchmark.initialize(v_lp0, pool.total_risky, s_star[0], lam)

    n = len(s_star)
    records = []

    for i, s_ext in enumerate(s_star):
        # Arbitrage step: bring pool price into no-arb band
        if i > 0:
            q_old = pool.total_risky
            pool.arbitrage_step(s_ext)
            q_new = pool.total_risky
            # Benchmark rebalances to match new Q
            bench.rebalance(q_new, s_ext)

        v_lp = pool.liquidation_value(s_ext)
        v_r = bench.liquidation_value(s_ext)
        records.append({
            "t": i,
            "s_star": s_ext,
            "s_pool": pool.pool_price,
            "X": pool.x,
            "Y": pool.y,
            "Xf": pool.xf,
            "Yf": pool.yf,
            "Q": pool.total_risky,
            "R": pool.total_numeraire,
            "V_LP": v_lp,
            "V_R": v_r,
            "LVR_tc": v_r - v_lp,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Simulation 2: Retained-fee pool with inventory schedule
# ---------------------------------------------------------------------------

def simulate_retained_fee_pool(
    s_star: np.ndarray,
    c: float,
    beta: float,
    tau: float,
    lam: float,
    s0: Optional[float] = None,
) -> pd.DataFrame:
    """Simulate a retained-fee pool whose inventory follows q(S) = c * S^(-beta).

    Uses the reserve curve f from Proposition 5.1 and Proposition 5.2 dynamics:
      - X(t) = q(S(t)) always (inventory schedule maintained exactly).
      - K = Y - f(X) is the non-decreasing fee buffer.
      - V_LP = Y + S*(1-lambda)*X (no separate fee accounts).
      - Benchmark V_R matches Q = X.
      - LVR_tc = V_R - V_LP should be ≈ 0 (Proposition 5.2).

    Parameters
    ----------
    s_star : external price path
    c      : inventory scale
    beta   : inventory exponent in (0, 1)
    tau    : pool fee rate
    lam    : external TC lambda
    s0     : optional override for initial internal price (default: s_star[0])

    Returns
    -------
    DataFrame with columns:
        t, s_star, s_pool, X, Y, K_buffer,
        V_LP, V_R, LVR_tc, schedule_error
    """
    if s0 is None:
        s0 = s_star[0]

    x0 = inventory_schedule_quantity(s0, c, beta)
    y0 = inventory_schedule_f(x0, c, beta)   # start with zero buffer
    pool = RetainedFeePool(x=x0, y=y0, c=c, beta=beta, tau=tau, lam=lam)

    v_lp0 = pool.liquidation_value(s_star[0])
    bench = RebalancingBenchmark.initialize(v_lp0, pool.x, s_star[0], lam)

    records = []
    for i, s_ext in enumerate(s_star):
        if i > 0:
            q_old = pool.x
            pool.arbitrage_step(s_ext)
            q_new = pool.x
            bench.rebalance(q_new, s_ext)

        v_lp = pool.liquidation_value(s_ext)
        v_r = bench.liquidation_value(s_ext)
        q_target = inventory_schedule_quantity(pool.pool_price, c, beta)
        schedule_err = pool.x - q_target  # should be ~0

        records.append({
            "t": i,
            "s_star": s_ext,
            "s_pool": pool.pool_price,
            "X": pool.x,
            "Y": pool.y,
            "K_buffer": pool.fee_buffer,
            "V_LP": v_lp,
            "V_R": v_r,
            "LVR_tc": v_r - v_lp,
            "schedule_error": schedule_err,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Simulation 3: GMM pool – no-trade zone
# ---------------------------------------------------------------------------

def simulate_gmm_pool(
    s_star: np.ndarray,
    pi_minus: float,
    pi_plus: float,
    lam: float,
    initial_wealth: float = 1.0,
) -> pd.DataFrame:
    """Simulate a GMM pool implementing Black-Scholes no-trade boundaries.

    Maps (pi_minus, pi_plus, lambda) -> GMM parameters (k, tau) via Section 5.3:
        a = pi_minus/(1-pi_minus),  b = pi_plus/(1-pi_plus)
        k = sqrt(a*b),  gamma = sqrt(a/b)
        tau = 1 - gamma/(1-lambda)

    Initializes pool at price S*(0) with risky fraction (pi_minus + pi_plus)/2.
    Verifies pi stays in [pi_minus, pi_plus] throughout.

    Parameters
    ----------
    s_star         : external price path
    pi_minus       : lower no-trade risky fraction
    pi_plus        : upper no-trade risky fraction
    lam            : external TC lambda
    initial_wealth : total initial wealth W = Y + S*(0)*X

    Returns
    -------
    DataFrame with columns:
        t, s_star, s_pool, X, Y, pi, V_LP, V_R, LVR_tc,
        pi_in_band
    """
    params = gmm_params_from_boundaries(pi_minus, pi_plus, lam)
    k = params["k"]
    tau = params["tau"]
    gamma = params["gamma"]

    # Initialise reserves at midpoint fraction pi_0 = sqrt(pi_minus * pi_plus)
    pi_0 = np.sqrt(pi_minus * pi_plus)
    # At S*(0): pi_0 = S*X/(Y + S*X)  and  Y = S*X*(k-1)*?
    # For GMM at internal price p=S*(0): X = (Ck/p)^{1/(k+1)}, Y=pX/k
    # We fix W = Y + S*X = S*X*(1 + 1/k) ... wait that's only at equilibrium.
    # Instead: choose X_0 such that  pi_0 = s0*X0/(Y0 + s0*X0)
    # and GMM invariant: Y0 = S0*X0*(k-1)?  No.
    # For GMM pool at price p=S*(0), after arb: Y = p*X/k  (from S = k*Y/X).
    # But we want pi_0 at S*(0): pi_0 = s0*X0 / (Y0 + s0*X0)
    # At GMM equilibrium (pool price = s0): Y0 = s0*X0/k
    # pi_eq = s0*X0 / (s0*X0/k + s0*X0) = k/(1+k)
    # So equilibrium pi is k/(1+k). We initialize there.
    s0 = s_star[0]
    # wealth = Y0 + s0*X0 = s0*X0/k + s0*X0 = s0*X0*(1+k)/k
    # X0 = wealth * k / (s0 * (1+k))
    x0 = initial_wealth * k / (s0 * (1 + k))
    y0 = s0 * x0 / k
    c_inv = x0**k * y0

    pool = GMMPool(
        x=x0, y=y0, c_inv=c_inv, k=k, tau=tau, lam=lam,
        pi_minus=pi_minus, pi_plus=pi_plus,
    )

    v_lp0 = pool.liquidation_value(s_star[0])
    bench = RebalancingBenchmark.initialize(v_lp0, pool.x, s_star[0], lam)

    records = []
    for i, s_ext in enumerate(s_star):
        if i > 0:
            q_old = pool.x
            pool.arbitrage_step(s_ext)
            q_new = pool.x
            bench.rebalance(q_new, s_ext)

        pi_t = pool.risky_fraction(s_ext)
        v_lp = pool.liquidation_value(s_ext)
        v_r = bench.liquidation_value(s_ext)

        records.append({
            "t": i,
            "s_star": s_ext,
            "s_pool": pool.pool_price,
            "X": pool.x,
            "Y": pool.y,
            "pi": pi_t,
            "V_LP": v_lp,
            "V_R": v_r,
            "LVR_tc": v_r - v_lp,
            "pi_in_band": (pi_minus - 1e-8 <= pi_t <= pi_plus + 1e-8),
        })

    return pd.DataFrame(records)
