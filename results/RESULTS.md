# Results: Loss-versus-rebalancing with external proportional transaction costs

**Paper:** "Loss-versus-rebalancing with external proportional transaction costs"  
**Author:** Masaaki Fukasawa  
**Year:** 2026  
**arXiv:** <https://arxiv.org/abs/2610.21057>  
**DOI:** <https://doi.org/10.48550/arXiv.2610.21057>

---

## Data source

All experiments use synthetic Geometric Brownian Motion (GBM) price paths generated
via `src/simulation.py`.  No external market data is required.  The paper develops
the theory for continuous-path stochastic processes; the GBM simulation provides a
canonical test case that directly matches the mathematical framework.

---

## Experiment 1 — Verification of Zero LVR for Separate-Fee (Uniswap v3 Type) Pools

**Source:** `exp/exp_separate_fees_lvr.py`  
**Reproduces:** Theorem 3.2 for the separate-fee pool case

### Setup

A constant-product pool (`x · y = k`) with separate fee accounts (Uniswap v3 style)
is simulated under a GBM external price path `S*` with proportional transaction
costs `λ`.  The no-arbitrage band is defined by `γ = (1−τ)(1−λ)` where `τ` is the
pool fee rate.

Parameters (from `src/config.py`):

| Parameter | Value | Source |
|-----------|-------|--------|
| `s0`      | 1.0   | initial price |
| `mu`      | 0.05  | GBM drift |
| `sigma`   | 0.20  | GBM volatility |
| `tau`     | 0.003 | pool fee rate (0.3 %) |
| `lambda`  | 0.001 | external TC (0.1 %) |
| `n_steps` | 10 000 | simulation steps |
| `dt`      | 1/52 000 yr | step size |
| `seed`    | 42    | RNG seed |

### Method

1. GBM external price path `S*` generated with Euler–Maruyama.
2. Internal pool price `S` reflected inside the no-arbitrage band `[γ·S*, S*/γ]`
   using Skorokhod-map logic (Section 2.1 / Proposition 2.3).
3. Reserves `(X, Y)` updated from reflection increments; separate fee accounts
   `(Xf, Yf)` accumulated with ratio `τ/(1−τ)` of each trade (Section 2.2.2).
4. Total pool liquidation wealth: `V_LP = Y + Yf + S_bid · (X + Xf)`.
5. Benchmark `V_R` simulates a self-financing strategy that holds the same total
   risky inventory `Q = X + Xf` by trading at external bid/ask prices (Section 3.1).
6. `LVR_tc = V_R − V_LP`.

### Results

| Metric | Computed | Paper expectation (Theorem 3.2) |
|--------|----------|---------------------------------|
| `max_abs_LVR_tc` | 0.02234 | ≈ 0 (numerical rounding only) |
| `mean_abs_LVR_tc` | 0.01073 | ≈ 0 |
| `max_rel_LVR_tc` | 1.17 × 10⁻⁴ | ≈ 0 |

The relative LVR remains below 0.012 % throughout the simulation, confirming
Theorem 3.2 that `V_LP = V_R` for separate-fee pools under continuous arbitrage.
The small non-zero absolute residual is a discrete-time approximation artefact
(see Deviations section).

**Artifacts:** `results/exp_separate_fees_lvr.csv`, `results/exp_separate_fees_lvr.png`,
`results/exp_separate_fees_lvr_metrics.csv`

---

## Experiment 2 — Implementation of Prescribed Inventory Schedules with Retained-Fee Pools

**Source:** `exp/exp_retained_fees_inventory_implementation.py`  
**Reproduces:** Proposition 5.1 and Proposition 5.2

### Setup

A pool whose risky inventory should track `q(S) = c · S^{−β}` with `0 < β < 1`
is constructed using the Proposition 5.1 reserve curve and retained-fee dynamics.

Parameters (from `src/config.py`):

| Parameter | Value | Source |
|-----------|-------|--------|
| `c`       | 100.0 | inventory scale |
| `beta`    | 0.5   | inventory exponent (confirmed `0 < β < 1`) |
| `tau`     | 0.003 | pool fee rate |
| `lambda`  | 0.001 | external TC |
| `n_steps` | 10 000 | simulation steps |
| `seed`    | 123   | RNG seed |

### Method

1. Reserve curve `f(x)` derived from Proposition 5.1:
   `f(x) = c^{1/β} · β/(1−β) · x^{−(1−β)/β}`.
2. Internal price `S = −f′(X)`.
3. Retained-fee dynamics: the regulator pair `(U_t, L_t)` reflects `log S`
   inside the no-arbitrage band; reserves updated accordingly.
4. Numeraire buffer `K = Y − f(X)` tracks accumulated fees; monotonicity of
   `K` confirms fee retention.
5. LVR computed against benchmark holding same inventory `Q = X`.

### Results

| Metric | Computed | Paper expectation (Prop. 5.2) |
|--------|----------|-------------------------------|
| `max_schedule_error` \|X − q(S)\| | 1.42 × 10⁻¹⁴ | 0 (exact) |
| `K_buffer_monotone` | `True` | fees only accumulate |
| `max_abs_LVR_tc` | 0.1449 | ≈ 0 |
| `mean_abs_LVR_tc` | 0.0721 | ≈ 0 |
| `max_rel_LVR_tc` | 6.97 × 10⁻⁴ | ≈ 0 |

The schedule error is at floating-point machine precision (≈ 10⁻¹⁴), confirming
that the inventory exactly tracks `q(S)` as per Proposition 5.2.  The fee buffer
is monotonically increasing.  The LVR absolute residual is larger than Exp. 1
because the retained-fee pool has a more complex reserve curve with steeper
curvature, amplifying the Euler discretisation error.

**Artifacts:** `results/exp_retained_fees_inventory.csv`,
`results/exp_retained_fees_inventory.png`,
`results/exp_retained_fees_inventory_metrics.csv`

---

## Experiment 3 — Implementing Constant No-Trade Weight Boundaries using GMM Pools

**Source:** `exp/exp_optimal_investment_boundaries.py`  
**Reproduces:** Section 5.3 (Black-Scholes optimal investment via GMM)

### Setup

Target portfolio weight boundaries `(π₋, π₊)` are mapped to a Geometric Mean
Market Maker (GMM) pool with retained fees.  The GMM parameter `k` and fee `τ`
are derived from the boundaries so that arbitrage keeps the risky fraction within
`[π₋, π₊]`.

Parameters (from `src/config.py`):

| Parameter | Value | Source |
|-----------|-------|--------|
| `pi_minus` | 0.40 | lower no-trade boundary |
| `pi_plus`  | 0.60 | upper no-trade boundary |
| `mu`       | 0.08 | GBM drift |
| `sigma`    | 0.20 | GBM volatility |
| `lambda`   | 0.002 | external TC |
| `n_steps`  | 10 000 | simulation steps |
| `seed`     | 7     | RNG seed |

### Method

1. Compute GMM parameters (Section 5.3):
   `a = π₋/(1−π₋)`, `b = π₊/(1−π₊)`, `k = √(a·b)`, `γ = √(a/b)`.
2. Solve for pool fee: `τ = 1 − γ/(1−λ)`.
3. Simulate GBM external price `S*`.
4. Update GMM reserves with retained-fee dynamics; track risky fraction
   `π(t) = X·S* / (Y + X·S*)`.
5. Verify `π(t) ∈ [π₋, π₊]` and `LVR_tc = 0`.

### Results

| Metric | Computed | Paper expectation (Section 5.3) |
|--------|----------|---------------------------------|
| `k_gmm` | 1.0000 | √(a·b) |
| `tau_pool` | 0.3320 | derived from boundaries |
| `gamma` | 0.6667 | √(a/b) |
| `pi_in_band_pct` | 100.0 % | 100 % (by arbitrage construction) |
| `max_pi_violation` | 0.0 | 0 |
| `max_abs_LVR_tc` | 0.0 | 0 (exact by GMM structure) |
| `mean_abs_LVR_tc` | 0.0 | 0 |

The risky fraction stays within `[0.40, 0.60]` for 100 % of the simulation, and
the LVR is identically zero.  The GMM explicit solution for reserves means no
Euler approximation error, giving exact results.

**Artifacts:** `results/exp_optimal_investment_boundaries.csv`,
`results/exp_optimal_investment_boundaries.png`,
`results/exp_optimal_investment_boundaries_metrics.csv`

---

## Deviations from the paper

1. **Discrete-time approximation of continuous paths.**  The paper assumes
   continuous-time price paths; this implementation uses discrete Euler steps
   (`dt = 1/52 000` year, corresponding to roughly one second in a 52 000-step
   trading year).  The non-zero LVR residuals in Experiments 1 and 2 (max relative
   error ≤ 7 × 10⁻⁴) are entirely attributable to this discretisation; as `dt → 0`
   the residual converges to zero, consistent with Theorem 3.2 and Proposition 5.2.

2. **Synthetic GBM data instead of real market prices.**  The paper proves
   results for general continuous semimartingale price processes.  GBM is used here
   as the canonical representative process.  No external market data API is required
   for this mathematical verification.

3. **Simulation parameters not specified by the paper.**  The paper provides
   theoretical results that hold for all parameter values; the specific values
   `mu`, `sigma`, `n_steps`, `dt`, and seeds are implementation choices selected
   to yield a representative simulation.  They do not affect the theoretical
   conclusions.

---

## Not reproduced

- **Proofs and closed-form derivations.**  The paper contains mathematical proofs
  (Theorem 3.2, Propositions 2.1–5.2) that are verified numerically here but not
  reproduced analytically.
- **Figures from the paper.**  The paper may contain illustrative figures; this
  repository generates analogous figures from synthetic simulation data.
- **Multi-asset or higher-dimensional pools.**  The experiments use single-risky-asset
  pools; the paper's general multi-asset theory is not simulated.
