"""
Shared experiment configuration.

All paper-derived parameters live here. Experiment scripts import from this
module so that a single change propagates to all experiments.
"""

# ── Experiment 1: Separate-fee pool (Theorem 3.2) ───────────────────────────
EXP_SEPARATE_FEES = {
    "s0": 1.0,          # initial external price
    "mu": 0.05,         # GBM drift
    "sigma": 0.20,      # GBM volatility
    "dt": 1 / 52_000,   # time step (≈ 1 second in a 52 000-step trading year)
    "n_steps": 10_000,  # number of steps
    "seed": 42,
    "x0": 100.0,        # initial risky reserve
    "y0": 100.0,        # initial numeraire reserve  →  S0 = Y/X = 1
    "tau": 0.003,       # pool fee rate (0.3%)
    "lam": 0.001,       # external proportional TC lambda (0.1%)
}

# ── Experiment 2: Retained-fee pool, inventory schedule (Proposition 5.2) ───
EXP_RETAINED_FEES = {
    "s0": 1.0,
    "mu": 0.05,
    "sigma": 0.20,
    "dt": 1 / 52_000,
    "n_steps": 10_000,
    "seed": 123,
    "c": 100.0,         # inventory scale in q(S) = c * S^(-beta)
    "beta": 0.5,        # inventory exponent; must satisfy 0 < beta < 1
    "tau": 0.003,
    "lam": 0.001,
}

# ── Experiment 3: GMM pool, Black-Scholes no-trade zone (Section 5.3) ───────
EXP_GMM_BOUNDARIES = {
    "s0": 1.0,
    "mu": 0.08,
    "sigma": 0.20,
    "dt": 1 / 52_000,
    "n_steps": 10_000,
    "seed": 7,
    "pi_minus": 0.40,      # lower no-trade risky fraction
    "pi_plus": 0.60,       # upper no-trade risky fraction
    "lam": 0.002,          # external TC
    "initial_wealth": 1.0,
}
