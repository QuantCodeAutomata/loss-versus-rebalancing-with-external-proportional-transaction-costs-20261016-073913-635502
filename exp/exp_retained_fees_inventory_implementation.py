"""
Experiment: Implementation of Prescribed Inventory Schedules with Retained-Fee Pools
=====================================================================================

Constructs and tests a pool that implements a specific target inventory schedule
q(S) = c * S^(-beta) using the retained-fee utility-indifference mechanism.

Methodology (Section 5.1, Proposition 5.1/5.2):
  1. Choose q(S) = c * S^(-beta), 0 < beta < 1.
  2. Define reserve curve f from Proposition 5.1:
       f(x) = c^{1/beta} * beta/(1-beta) * x^{-(1-beta)/beta}
     satisfying -f'(x) = S when x = q(S).
  3. Initialise pool at zero fee buffer (K=0), so Y_0 = f(X_0).
  4. Simulate retained-fee dynamics: at each arbitrage event, adjust X to q(S_new)
     and update Y so that K = Y - f(X) is preserved (never decreases).
  5. Verify X(t) = q(S(t)) exactly at all times.
  6. Verify LVR_tc = V_R - V_LP ≈ 0 (Proposition 5.2).

Expected outcomes:
  - schedule_error ≈ 0 throughout (machine precision).
  - |LVR_tc| numerically ~0 (within discrete-time rounding).
  - K(t) non-decreasing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation import generate_gbm, simulate_retained_fee_pool


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
PARAMS = {
    # GBM
    "s0": 1.0,
    "mu": 0.05,
    "sigma": 0.20,
    "dt": 1 / 52_000,
    "n_steps": 10_000,
    "seed": 123,
    # Inventory schedule q(S) = c * S^(-beta)
    "c": 100.0,
    "beta": 0.5,       # beta in (0, 1)
    # Pool fee and external TC
    "tau": 0.003,
    "lam": 0.001,
}


def run(params: dict = None) -> dict:
    """Run the retained-fee inventory-schedule experiment.

    Returns
    -------
    dict with keys:
        df              : simulation DataFrame
        max_schedule_err: max |schedule_error| = max |X - q(S)|
        max_lvr         : max |LVR_tc|
        mean_lvr        : mean |LVR_tc|
        max_rel         : max |LVR_tc| / V_LP
        k_monotone      : True if K(t) is non-decreasing throughout
        params          : parameters used
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

    df = simulate_retained_fee_pool(
        s_star=s_star,
        c=params["c"],
        beta=params["beta"],
        tau=params["tau"],
        lam=params["lam"],
    )

    abs_lvr = df["LVR_tc"].abs()
    k_buf = df["K_buffer"].values
    k_monotone = bool(np.all(np.diff(k_buf) >= -1e-12))

    return {
        "df": df,
        "max_schedule_err": df["schedule_error"].abs().max(),
        "max_lvr": abs_lvr.max(),
        "mean_lvr": abs_lvr.mean(),
        "max_rel": (abs_lvr / df["V_LP"]).max(),
        "k_monotone": k_monotone,
        "params": params,
    }


def save_results(results: dict, output_dir: str = "results") -> None:
    """Save metrics CSV and plots to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    df = results["df"]

    df.iloc[::10].to_csv(
        os.path.join(output_dir, "exp_retained_fees_inventory.csv"), index=False
    )

    metrics = {
        "max_schedule_error": results["max_schedule_err"],
        "max_abs_LVR_tc": results["max_lvr"],
        "mean_abs_LVR_tc": results["mean_lvr"],
        "max_rel_LVR_tc": results["max_rel"],
        "K_buffer_monotone": results["k_monotone"],
    }
    pd.Series(metrics).to_csv(
        os.path.join(output_dir, "exp_retained_fees_inventory_metrics.csv"), header=True
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(df["t"], df["s_star"], label="S* (external)", linewidth=0.8)
    axes[0].plot(df["t"], df["s_pool"], label="S (pool)", linewidth=0.8, alpha=0.7)
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Retained-Fee Pool: Inventory Schedule q(S) = c·S^{-β}")

    axes[1].plot(df["t"], df["X"], label="X (actual)", linewidth=0.8)
    c = results["params"]["c"]
    beta = results["params"]["beta"]
    q_target = c * df["s_pool"] ** (-beta)
    axes[1].plot(df["t"], q_target, label="q(S) (target)", linewidth=0.8,
                 linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Risky inventory X")
    axes[1].legend(fontsize=8)

    axes[2].plot(df["t"], df["K_buffer"], color="green", linewidth=0.8)
    axes[2].set_ylabel("K = Y − f(X)\n(fee buffer)")

    axes[3].plot(df["t"], df["LVR_tc"], color="red", linewidth=0.8)
    axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[3].set_ylabel("LVR_tc = V_R − V_LP")
    axes[3].set_xlabel("Time step")
    axes[3].set_title(
        f"max|LVR_tc|={results['max_lvr']:.2e}, "
        f"max|sched_err|={results['max_schedule_err']:.2e}, "
        f"K monotone={results['k_monotone']}"
    )

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "exp_retained_fees_inventory.png"), dpi=150
    )
    plt.close(fig)

    print(f"[exp_retained_fees] max|schedule_error| = {results['max_schedule_err']:.6e}")
    print(f"[exp_retained_fees] max|LVR_tc| = {results['max_lvr']:.6e}")
    print(f"[exp_retained_fees] K_buffer monotone = {results['k_monotone']}")


if __name__ == "__main__":
    print("Running: Retained-Fee Pool Inventory Schedule (Proposition 5.2)")
    results = run()
    save_results(results)
    print("Done.")
