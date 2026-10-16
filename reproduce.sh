#!/usr/bin/env bash
# reproduce.sh — Run all experiments and tests for the LVR replication.
# Usage:  bash reproduce.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo " LVR Replication – reproduce.sh"
echo "=========================================="

# ── 1. Install dependencies ──────────────────────────────────────────────────
echo ""
echo "[1/4] Installing dependencies…"
pip install -q -r requirements.txt

# ── 2. Run test suite ────────────────────────────────────────────────────────
echo ""
echo "[2/4] Running pytest…"
python -m pytest tests/ -v --tb=short

# ── 3. Run experiments ───────────────────────────────────────────────────────
echo ""
echo "[3/4] Running experiments…"

mkdir -p results

echo "  → exp_separate_fees_lvr"
python exp/exp_separate_fees_lvr.py

echo "  → exp_retained_fees_inventory_implementation"
python exp/exp_retained_fees_inventory_implementation.py

echo "  → exp_optimal_investment_boundaries"
python exp/exp_optimal_investment_boundaries.py

# ── 4. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Done.  Results saved to results/"
echo "  Files:"
ls -1 results/
echo ""
echo "See results/RESULTS.md for a comparison with the paper's expected values."
