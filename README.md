# Loss-versus-rebalancing with external proportional transaction costs

**Paper:** "Loss-versus-rebalancing with external proportional transaction costs"  
**Author:** Masaaki Fukasawa  
**Year:** 2026  
**arXiv:** <https://arxiv.org/abs/2610.21057>  
**DOI:** <https://doi.org/10.48550/arXiv.2610.21057>

---

## Overview

This repository replicates three experiments from the paper. The paper develops a
framework for analyzing Loss-Versus-Rebalancing (LVR) for Automated Market Maker
(AMM) pools in the presence of *external* proportional transaction costs λ applied
to the benchmark rebalancing strategy.

The central finding (Theorem 3.2) is that the LVR relative to the external-TC
benchmark is **zero** for any reserve-curve pool with pure continuous arbitrage,
provided both the pool and the benchmark face the same transaction costs.

---

## Repository Structure

```
.
├── data/            # (empty – all data is synthetically generated via GBM)
├── src/
│   ├── __init__.py
│   ├── config.py          # Paper-derived parameters (shared by all experiments)
│   ├── pool_dynamics.py   # AMM pool math: constant-product, retained-fee, GMM
│   ├── benchmark.py       # Self-financing rebalancing benchmark V_R
│   └── simulation.py      # GBM path generator + three simulation orchestrators
├── exp/
│   ├── exp_separate_fees_lvr.py
│   ├── exp_retained_fees_inventory_implementation.py
│   └── exp_optimal_investment_boundaries.py
├── tests/
│   ├── test_pool_dynamics.py
│   └── test_simulations.py
├── results/
│   ├── RESULTS.md
│   └── *.png / *.csv
├── reproduce.sh
├── requirements.txt
└── .env.example
```

---

## Experiments

### 1 · Verification of Zero LVR for Separate-Fee Pools (Theorem 3.2)

`exp/exp_separate_fees_lvr.py`

Simulates a constant-product pool with separate fee accounts (Uniswap v3 style)
under a GBM external price path with proportional transaction costs.  Verifies
that `LVR_tc = V_R − V_LP ≈ 0` throughout, confirming Theorem 3.2.

### 2 · Inventory Schedule Implementation with Retained Fees (Proposition 5.2)

`exp/exp_retained_fees_inventory_implementation.py`

Constructs a pool whose risky inventory follows `q(S) = c · S^{−β}` using the
Proposition 5.1 reserve curve and retained-fee dynamics.  Verifies that `X(t) = q(S(t))`
exactly and `LVR_tc ≈ 0`.

### 3 · No-Trade Zone via Geometric Mean Pool (Section 5.3)

`exp/exp_optimal_investment_boundaries.py`

Maps Black-Scholes optimal investment no-trade boundaries `(π₋, π₊)` to a GMM
pool with retained fees.  Verifies that the risky fraction `π(t)` stays within
`[π₋, π₊]` and `LVR_tc ≈ 0`.

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments and tests
bash reproduce.sh
```

All experiments use synthetic GBM price paths; no external data API is required.

---

## Parameters

All paper-derived parameters are in `src/config.py`.  Each experiment imports
its parameter dictionary from that module.

---

## Results

See `results/RESULTS.md` for quantitative results and comparison with the paper's
theoretical predictions.
