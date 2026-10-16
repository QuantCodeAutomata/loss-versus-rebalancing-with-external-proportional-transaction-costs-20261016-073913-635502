"""
Experiment: Verification of Zero LVR for Separate-Fee (Uniswap v3 Type) Pools
==============================================================================

Verifies Theorem 3.2: for a reserve-curve pool with *separate* fee accounts
under continuous external price dynamics with proportional transaction costs,
the Loss-Versus-Rebalancing relative to the external benchmark is exactly zero:

    LVR_tc = V_R - V_LP = 0

Methodology (Section 2.2.2 and Theorem 3.2):
  1. External price S* follows GBM.
  2. Constant-product pool (f(x) = C/x) with no-arb band [gamma*S*, S*/gamma],
     gamma = (1 - tau)(1 - lambda).
  3. Separate fee accounts: dYf = tau/(1-tau)*dY_up (price rises),
                             dXf = tau/(1-tau)*dX_up (price falls).
  4. Self-financing benchmark V_R matches Q = X + Xf, trading at external bid/ask.
  5. Both V_LP and V_R liquidated at S*(1 - lambda).
  6. LVR_tc = V_R_liq - V_LP_liq is recorded at each step.

Expected outcome: |LVR_tc| numerically ~0 (within discrete-time rounding errors
that vanish as dt -> 0).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation import generate_gbm, simulate_separate_fee_pool


# ---------------------------------------------------------------------------
# Experiment parameters (paper-derived)
# ---------------------------------------------------------------------------
PARAMS = {
    # GBM
    "s0": 1.0,
    "mu": 0.05,
    "sigma": 0.20,
    "dt": 1 / 52_000,   # very fine grid → small discrete error
    "n_steps": 10_000,
    "seed": 42,
    # Pool
    "x0": 100.0,
    "y0": 100.0,        # initial price S = Y/X = 1.0
    "tau": 0.003,       # 0.3% fee (Uniswap v3 tier)
    "lam": 0.001,       # 0.1% external TC
}


def run(params: dict = None) -> dict:
    """Run the separate-fee LVR verification experiment.

    Returns
    -------
    dict with keys:
        df        : full simulation DataFrame
        max_lvr   : max |LVR_tc| observed
        mean_lvr  : mean |LVR_tc| observed
        max_rel   : max |LVR_tc| / V_LP (relative error)
        params    : experiment parameters used
    """
    if params is None:
        params = PARAMS.copy()

    s_star = generate_gbm(
        s0=params["s0"],
        mu=params["mu"],
        sigma=params["sigma"],
        dt=params["dt"],
        n_steps=params["n_steps"],
        seed=params["seed"],
    )

    df = simulate_separate_fee_pool(
        s_star=s_star,
        x0=params["x0"],
        y0=params["y0"],
        tau=params["tau"],
        lam=params["lam"],
    )

    abs_lvr = df["LVR_tc"].abs()
    max_lvr = abs_lvr.max()
    mean_lvr = abs_lvr.mean()
    max_rel = (abs_lvr / df["V_LP"]).max()

    return {
        "df": df,
        "max_lvr": max_lvr,
        "mean_lvr": mean_lvr,
        "max_rel": max_rel,
        "params": params,
    }


def save_results(results: dict, output_dir: str = "results") -> None:
    """Save metrics CSV and plot to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    df = results["df"]

    # Save CSV (sample every 10 rows to keep file small)
    df.iloc[::10].to_csv(
        os.path.join(output_dir, "exp_separate_fees_lvr.csv"), index=False
    )

    # Save metrics
    metrics = {
        "max_abs_LVR_tc": results["max_lvr"],
        "mean_abs_LVR_tc": results["mean_lvr"],
        "max_rel_LVR_tc": results["max_rel"],
    }
    pd.Series(metrics).to_csv(
        os.path.join(output_dir, "exp_separate_fees_lvr_metrics.csv"), header=True
    )

    # --- Plot 1: LVR_tc over time ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(df["t"], df["s_star"], label="S* (external)", linewidth=0.8)
    axes[0].plot(df["t"], df["s_pool"], label="S (pool)", linewidth=0.8, alpha=0.7)
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)
    axes[0].set_title("External vs Pool Price (Separate-Fee Pool)")

    axes[1].plot(df["t"], df["V_LP"], label="V_LP", linewidth=0.8)
    axes[1].plot(df["t"], df["V_R"], label="V_R", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Wealth (liquidation)")
    axes[1].legend(fontsize=8)

    axes[2].plot(df["t"], df["LVR_tc"], color="red", linewidth=0.8)
    axes[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("LVR_tc = V_R − V_LP")
    axes[2].set_xlabel("Time step")
    axes[2].set_title(
        f"Theorem 3.2 Verification: max|LVR_tc|={results['max_lvr']:.2e}, "
        f"max_rel={results['max_rel']:.2e}"
    )

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "exp_separate_fees_lvr.png"), dpi=150
    )
    plt.close(fig)

    print(f"[exp_separate_fees_lvr] max|LVR_tc| = {results['max_lvr']:.6e}")
    print(f"[exp_separate_fees_lvr] mean|LVR_tc| = {results['mean_lvr']:.6e}")
    print(f"[exp_separate_fees_lvr] max_rel = {results['max_rel']:.6e}")


if __name__ == "__main__":
    print("Running: Verification of Zero LVR for Separate-Fee Pools (Theorem 3.2)")
    results = run()
    save_results(results)
    print("Done.")
