"""
Self-financing rebalancing benchmark strategy V_R.

The benchmark mirrors the pool's total risky inventory Q(t) = X(t) + Xf(t),
trading at external bid/ask prices.  Theorem 3.2 asserts V_R = V_LP for the
separate-fee pool; Propositions 5.2/5.3 assert V_R = V_LP for the retained-fee
and GMM pools.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class RebalancingBenchmark:
    """Self-financing strategy that tracks a target risky quantity Q(t).

    The benchmark holds Q(t) risky units and adjusts numeraire cash N(t) to
    remain self-financing.  Trades execute at external bid/ask prices:
        ask price (buying risky)  : S* / (1 - lambda)
        bid price (selling risky) : S* * (1 - lambda)

    Initial state matches the pool's liquidation wealth:
        V_R(0) = V_LP(0)  ==>  N(0) = V_LP(0) - S*(0)*(1-lambda)*Q(0)

    Parameters
    ----------
    n_cash   : current numeraire cash balance N(t)
    q        : current risky holding Q(t)
    lam      : external proportional transaction cost lambda
    """
    n_cash: float   # numeraire cash N(t)
    q: float        # risky holding Q(t)
    lam: float      # external TC lambda

    @classmethod
    def initialize(
        cls,
        initial_wealth_liq: float,
        q0: float,
        s_ext0: float,
        lam: float,
    ) -> "RebalancingBenchmark":
        """Initialize benchmark to match pool's initial liquidation value.

        Parameters
        ----------
        initial_wealth_liq : V_LP(0) = pool's liquidation wealth at t=0
        q0                 : initial pool total risky Q(0)
        s_ext0             : external price S*(0)
        lam                : external TC lambda
        """
        # n_cash = V_LP(0) - S*(0)*(1-lambda)*Q(0)
        n_cash = initial_wealth_liq - s_ext0 * (1 - lam) * q0
        return cls(n_cash=n_cash, q=q0, lam=lam)

    def rebalance(self, q_new: float, s_ext: float) -> float:
        """Adjust risky holding to q_new, paying/receiving at bid or ask.

        Trade cost:
          - Buy (dQ > 0): pay S* / (1 - lambda) per unit
          - Sell (dQ < 0): receive S* * (1 - lambda) per unit

        Self-financing constraint:
            dN = - execution_price * dQ

        Parameters
        ----------
        q_new  : new target risky quantity
        s_ext  : current external price S*

        Returns
        -------
        execution_cost : total cost of rebalancing (positive = cash paid out)
        """
        dq = q_new - self.q
        if dq > 0:
            execution_price = s_ext / (1 - self.lam)   # buying at ask
        elif dq < 0:
            execution_price = s_ext * (1 - self.lam)   # selling at bid
        else:
            execution_price = s_ext

        cost = execution_price * dq
        self.n_cash -= cost
        self.q = q_new
        return cost

    def liquidation_value(self, s_ext: float) -> float:
        """Benchmark liquidation wealth: sell Q at external bid S*(1-lambda).

        V_R = N + S*(1 - lambda) * Q.
        """
        return self.n_cash + s_ext * (1 - self.lam) * self.q

    def mark_to_market(self, s_ext: float) -> float:
        """Mid-market value (informational): N + S* * Q."""
        return self.n_cash + s_ext * self.q
