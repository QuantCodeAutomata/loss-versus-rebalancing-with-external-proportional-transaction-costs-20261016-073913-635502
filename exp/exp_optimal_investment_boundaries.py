"""
Experiment: Implementing Constant No-Trade Weight Boundaries using Geometric-Mean Pools
========================================================================================

Maps Black-Scholes optimal investment no-trade zone boundaries (pi_minus, pi_plus)
to a Geometric Mean Market Maker (GMM) pool with retained fees, and verifies that
the pool's risky fraction pi(t) = S*(t)*X(t)/(Y(t)+S*(t)*X(t)) stays within
[pi_minus, pi_plus] throughout the simulation.

Methodology (Section 5.3):
  1. Given 0 < pi_minus < pi_plus < 1 and external TC lambda, derive GMM params:
       a = pi_minus/(1-pi_minus),  b = pi_plus/(1-pi_plus)
       k = sqrt(a*b),  gamma = sqrt(a/b)
       tau = 1 - gamma/(1-lambda)
  2. Initialise GMM pool: X^k * Y = C.  Start at equilibrium fraction k/(1+k).
  3. Simulate external GBM price S*.  At each step apply Skorokhod reflection in
     [gamma*S*, S*/gamma] by adjusting (X,Y) along the GMM invariant.
  4. Record pi(t) and verify it stays in [pi_minus, pi_plus].
  5. Track V_LP = Y + S*(1-lambda)*X and benchmark V_R; verify LVR_tc ≈ 0.

Expected outcomes:
  - pi(t) in [pi_minus, pi_plus] at all times (within numerical tolerance).
  - |LVR_tc| numerically ~0.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pool_dynamics import gmm_params_from_boundaries
from src.simulation import generate_gbm, simulate_gmm_pool


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
PARAMS = {
    # GBM (Black-Scholes external price model, Section 5.3)
    "s0": 1.0,
    "mu": 0.08,
    "sigma": 0.20,
    "dt": 1 / 52_000,
    "n_steps": 10_000,
    "seed": 7,
    # No-trade zone boundaries
    "pi_minus": 0.40,   # lower boundary of risky fraction
    "pi_plus": 0.60,    # upper boundary of risky fraction
    # External TC
    "lam": 0.002,
    # Initial total wealth W = Y(0) + S*(0)*X(0)
    "initial_wealth": 1.0,
}


def run(params: dict = None) -> dict:
    """Run the GMM no-trade-zone experiment.

    Returns
    -------
    dict with keys:
        df              : simulation DataFrame
        params_gmm      : derived GMM parameters (k, tau, gamma, a, b)
        pi_in_band_pct  : fraction of time steps where pi in [pi_minus, pi_plus]
        max_pi_violation: max overshoot of pi outside band
        max_lvr         : max |LVR_tc|
        max_rel         : max |LVR_tc| / V_LP
        params          : experiment parameters used
    """
    if params is None:
        params = PARAMS.copy()

    params_gmm = gmm_params_from_boundaries(
        pi_minus=params["pi_minus"],
        pi_plus=params["pi_plus"],
        lam=params["lam"],
    )

    s_star = generate_gbm(
        s0=params["s0"],
        mu=params["mu"],
        sigma=params["sigma"],
        dt=params["dt"],
        n_steps=params["n_steps"],
        seed=params["seed"],
    )

    df = simulate_gmm_pool(
        s_star=s_star,
        pi_minus=params["pi_minus"],
        pi_plus=params["pi_plus"],
        lam=params["lam"],
        initial_wealth=params["initial_wealth"],
    )

    pi_in_band_pct = df["pi_in_band"].mean() * 100.0

    # Max violation: how far pi goes outside [pi_minus, pi_plus]
    violations_lo = np.maximum(0, params["pi_minus"] - df["pi"].values)
    violations_hi = np.maximum(0, df["pi"].values - params["pi_plus"])
    max_pi_violation = float(np.maximum(violations_lo, violations_hi).max())

    abs_lvr = df["LVR_tc"].abs()

    return {
        "df": df,
        "params_gmm": params_gmm,
        "pi_in_band_pct": pi_in_band_pct,
        "max_pi_violation": max_pi_violation,
        "max_lvr": abs_lvr.max(),
        "mean_lvr": abs_lvr.mean(),
        "max_rel": (abs_lvr / df["V_LP"]).max(),
        "params": params,
    }


def save_results(results: dict, output_dir: str = "results") -> None:
    """Save metrics CSV and plots."""
    os.makedirs(output_dir, exist_ok=True)
    df = results["df"]
    p = results["params"]
    pg = results["params_gmm"]

    df.iloc[::10].to_csv(
        os.path.join(output_dir, "exp_optimal_investment_boundaries.csv"), index=False
    )

    metrics = {
        "k_gmm": pg["k"],
        "tau_pool": pg["tau"],
        "gamma": pg["gamma"],
        "pi_in_band_pct": results["pi_in_band_pct"],
        "max_pi_violation": results["max_pi_violation"],
        "max_abs_LVR_tc": results["max_lvr"],
        "mean_abs_LVR_tc": results["mean_lvr"],
        "max_rel_LVR_tc": results["max_rel"],
    }
    pd.Series(metrics).to_csv(
        os.path.join(output_dir, "exp_optimal_investment_boundaries_metrics.csv"),
        header=True,
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(df["t"], df["s_star"], linewidth=0.8, label="S* (GBM)")
    axes[0].set_ylabel("Price S*")
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        f"GMM No-Trade Zone: pi ∈ [{p['pi_minus']}, {p['pi_plus']}], "
        f"k={pg['k']:.3f}, tau={pg['tau']:.4f}"
    )

    axes[1].plot(df["t"], df["pi"], linewidth=0.8, label="pi(t)", color="steelblue")
    axes[1].axhline(p["pi_minus"], color="red", linestyle="--", linewidth=0.8,
                    label=f"pi_minus={p['pi_minus']}")
    axes[1].axhline(p["pi_plus"], color="green", linestyle="--", linewidth=0.8,
                    label=f"pi_plus={p['pi_plus']}")
    axes[1].set_ylabel("Risky fraction pi(t)")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(p["pi_minus"] - 0.05, p["pi_plus"] + 0.05)

    axes[2].plot(df["t"], df["LVR_tc"], color="red", linewidth=0.8)
    axes[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("LVR_tc = V_R − V_LP")
    axes[2].set_xlabel("Time step")
    axes[2].set_title(
        f"LVR_tc: max={results['max_lvr']:.2e}, "
        f"pi_in_band={results['pi_in_band_pct']:.1f}%"
    )

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "exp_optimal_investment_boundaries.png"), dpi=150
    )
    plt.close(fig)

    print(f"[exp_optimal_investment] GMM params: k={pg['k']:.4f}, tau={pg['tau']:.4f}, "
          f"gamma={pg['gamma']:.4f}")
    print(f"[exp_optimal_investment] pi_in_band = {results['pi_in_band_pct']:.2f}%")
    print(f"[exp_optimal_investment] max_pi_violation = {results['max_pi_violation']:.2e}")
    print(f"[exp_optimal_investment] max|LVR_tc| = {results['max_lvr']:.6e}")


if __name__ == "__main__":
    print("Running: GMM No-Trade Zone (Section 5.3)")
    results = run()
    save_results(results)
    print("Done.")
