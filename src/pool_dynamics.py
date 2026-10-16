"""
Core AMM pool dynamics for Loss-Versus-Rebalancing replication.

Implements:
  - Constant-product reserve curve (Section 2.1.1)
  - No-arbitrage band with external proportional transaction costs (Section 2.1)
  - Separate-fee pool dynamics (Section 2.2.2, Uniswap v3 type)
  - Retained-fee pool with inventory schedule (Section 2.2.3, Proposition 5.1/5.2)
  - Geometric Mean Market Maker (GMM) pool for no-trade zone (Section 5.3)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Reserve-curve helpers
# ---------------------------------------------------------------------------

def constant_product_price(x: float, k_const: float) -> float:
    """Pool price for constant-product curve  x * y = k_const.

    Price  S = -f'(x) = k_const / x^2  (Section 2.1.1).

    Parameters
    ----------
    x       : risky reserve
    k_const : invariant  x * y

    Returns
    -------
    Pool marginal price S = y / x = k_const / x^2.
    """
    return k_const / x**2


def constant_product_reserves(s: float, k_const: float) -> Tuple[float, float]:
    """Recover (x, y) from price s and constant-product invariant k_const.

    From  S = k_const / x^2:  x = sqrt(k_const / s),  y = sqrt(k_const * s).
    """
    x = np.sqrt(k_const / s)
    y = np.sqrt(k_const * s)
    return x, y


# Inventory-schedule reserve curve (Proposition 5.1)
def inventory_schedule_f(x: float, c: float, beta: float) -> float:
    """Reserve curve y = f(x) for target schedule q(S) = c * S^(-beta).

    Derived from Proposition 5.1:  f(q(p)) = integral_0^p q(r)dr - p * q(p).
    With q(r) = c * r^(-beta):
        integral_0^p c r^{-beta} dr = c p^{1-beta} / (1-beta)
        f(q(p)) = c p^{1-beta} / (1-beta) - p * c p^{-beta}
                = c p^{1-beta} * beta / (1-beta)

    Substituting p = (c/x)^{1/beta}  (inverse of q(p) = c p^{-beta} → x = c p^{-beta}):
        f(x) = c * beta/(1-beta) * (c/x)^{(1-beta)/beta}
             = c^{1/beta} * beta/(1-beta) * x^{-(1-beta)/beta}

    Parameters
    ----------
    x    : risky reserve
    c    : inventory scale parameter
    beta : inventory exponent in (0, 1)
    """
    assert 0 < beta < 1, "beta must be in (0, 1)"
    exponent = -(1 - beta) / beta
    return c ** (1 / beta) * beta / (1 - beta) * x ** exponent


def inventory_schedule_f_prime(x: float, c: float, beta: float) -> float:
    """First derivative f'(x) = -(c/x)^{1/beta}.

    At x = q(S) = c S^{-beta}: f'(x) = -S  (confirms S = -f'(x)).
    """
    return -((c / x) ** (1 / beta))


def inventory_schedule_f_double_prime(x: float, c: float, beta: float) -> float:
    """Second derivative f''(x) = (1/beta) * c^{1/beta} * x^{-1/beta - 1}."""
    return (1 / beta) * c ** (1 / beta) * x ** (-1 / beta - 1)


def inventory_schedule_price(x: float, c: float, beta: float) -> float:
    """Internal price S = -f'(x) = (c/x)^{1/beta}."""
    return (c / x) ** (1 / beta)


def inventory_schedule_quantity(s: float, c: float, beta: float) -> float:
    """Risky inventory from price: q(S) = c * S^(-beta)."""
    return c * s ** (-beta)


# GMM reserve curve helpers (Section 5.3)
def gmm_price(x: float, y: float, k: float) -> float:
    """GMM internal price  p = k * y / x  from invariant x^k * y = C.

    For x^k * y = C: f(x) = C / x^k, f'(x) = -k C / x^{k+1} = -k y / x,
    so price S = -f'(x) = k * y / x.
    """
    return k * y / x


def gmm_reserves_from_price(s: float, c_inv: float, k: float) -> Tuple[float, float]:
    """Recover (x, y) from GMM internal price s and invariant C = x^k * y.

    From  s = k y / x  and  x^k y = C:
        y = s x / k  →  x^k * s x / k = C  →  x^{k+1} = C k / s
        x = (C k / s)^{1/(k+1)},  y = s x / k
    """
    x = (c_inv * k / s) ** (1 / (k + 1))
    y = s * x / k
    return x, y


def risky_fraction(s_ext: float, x: float, y: float) -> float:
    """Risky fraction  π = S* X / (Y + S* X)  at external price s_ext."""
    return s_ext * x / (y + s_ext * x)


# ---------------------------------------------------------------------------
# Pool-state containers
# ---------------------------------------------------------------------------

@dataclass
class SeparateFeePool:
    """AMM pool with separate fee accounts (Uniswap v3 type).

    Attributes mirror Section 2.2.2 notation:
      x, y   : main risky and numeraire reserves
      xf, yf : fee-account holdings (risky and numeraire)
      k_const: constant-product invariant  x * y
      tau    : pool fee rate
      lam    : external proportional transaction cost  lambda
    """
    x: float          # main risky reserve X
    y: float          # main numeraire reserve Y
    xf: float         # risky fee account Xf
    yf: float         # numeraire fee account Yf
    k_const: float    # invariant X * Y
    tau: float        # pool fee  tau
    lam: float        # external TC  lambda

    @property
    def gamma(self) -> float:
        """No-arb band half-width  gamma = (1 - tau)(1 - lambda)."""
        return (1 - self.tau) * (1 - self.lam)

    @property
    def pool_price(self) -> float:
        """Current internal pool price S = Y / X."""
        return self.y / self.x

    @property
    def total_risky(self) -> float:
        """Total risky holdings Q = X + Xf."""
        return self.x + self.xf

    @property
    def total_numeraire(self) -> float:
        """Total numeraire holdings R = Y + Yf."""
        return self.y + self.yf

    def liquidation_value(self, s_ext: float) -> float:
        """Liquidation wealth at external bid price  S*(1 - lambda).

        V_LP = (Y + Yf) + S*(1 - lambda) * (X + Xf).
        """
        s_bid = s_ext * (1 - self.lam)
        return self.total_numeraire + s_bid * self.total_risky

    def no_arb_band(self, s_ext: float) -> Tuple[float, float]:
        """Return  (lower, upper)  of no-arbitrage band  [gamma*S*, S*/gamma].

        No arbitrage is profitable when  gamma*S* <= S <= S*/gamma.
        (Section 2.1, equation defining the band with gamma = (1-tau)(1-lambda).)
        """
        g = self.gamma
        return g * s_ext, s_ext / g

    def arbitrage_step(self, s_ext: float) -> Tuple[float, float, float, float]:
        """Apply one arbitrage step: clamp pool price to no-arb band.

        When S* shifts, check if the current pool price S is still within
        [gamma*S*, S*/gamma].  If not, execute arbitrage to bring S to the
        nearest boundary, updating reserves (X, Y) along the constant-product
        curve and accumulating fees in (Xf, Yf).

        Fee accumulation rules (Section 2.2.2):
          - When price rises (pool gives X, receives Y):  dYf = tau/(1-tau) * dY_up
          - When price falls (pool gives Y, receives X):  dXf = tau/(1-tau) * dX_up

        Returns
        -------
        (dx, dy, dxf, dyf) : changes in each account during this step.
        """
        lower, upper = self.no_arb_band(s_ext)
        s_curr = self.pool_price
        dx, dy, dxf, dyf = 0.0, 0.0, 0.0, 0.0

        if s_curr < lower:
            # Pool price too low; arbitrageur buys cheap X from pool,
            # driving pool price up to lower boundary gamma*S*.
            s_new = lower
            x_new, y_new = constant_product_reserves(s_new, self.k_const)
            dx = x_new - self.x   # negative (pool loses X)
            dy = y_new - self.y   # positive (pool gains Y)
            # Fee collected in numeraire (Y flowing in): dYf = tau/(1-tau) * dY_up
            dyf = (self.tau / (1 - self.tau)) * dy
            self.x, self.y, self.yf = x_new, y_new, self.yf + dyf

        elif s_curr > upper:
            # Pool price too high; arbitrageur sells X to pool,
            # driving pool price down to upper boundary S*/gamma.
            s_new = upper
            x_new, y_new = constant_product_reserves(s_new, self.k_const)
            dx = x_new - self.x   # positive (pool gains X)
            dy = y_new - self.y   # negative (pool pays Y)
            # Fee collected in risky (X flowing in): dXf = tau/(1-tau) * dX_up
            dxf = (self.tau / (1 - self.tau)) * dx
            self.x, self.y, self.xf = x_new, y_new, self.xf + dxf

        return dx, dy, dxf, dyf


@dataclass
class RetainedFeePool:
    """AMM pool with retained fees and inventory schedule q(S) = c * S^(-beta).

    Fees are retained in the numeraire buffer K = Y - f(X), which grows
    monotonically.  The pool's internal price S = (c/X)^{1/beta} tracks the
    external price S* through the no-arb band.
    (Section 2.2.3, Proposition 5.1/5.2.)

    Attributes
    ----------
    x      : risky reserve X  (always equals q(S))
    y      : numeraire reserve Y  (= f(X) + K, where K >= 0 is the buffer)
    c      : inventory schedule scale
    beta   : inventory schedule exponent in (0, 1)
    tau    : pool fee rate
    lam    : external TC lambda
    """
    x: float
    y: float
    c: float
    beta: float
    tau: float
    lam: float

    @property
    def gamma(self) -> float:
        return (1 - self.tau) * (1 - self.lam)

    @property
    def pool_price(self) -> float:
        """Internal price S = (c/X)^{1/beta} = -f'(X)."""
        return inventory_schedule_price(self.x, self.c, self.beta)

    @property
    def fee_buffer(self) -> float:
        """Numeraire buffer K = Y - f(X) (accumulated retained fees)."""
        return self.y - inventory_schedule_f(self.x, self.c, self.beta)

    def liquidation_value(self, s_ext: float) -> float:
        """V_LP = Y + S*(1 - lambda) * X."""
        return self.y + s_ext * (1 - self.lam) * self.x

    def no_arb_band(self, s_ext: float) -> Tuple[float, float]:
        g = self.gamma
        return g * s_ext, s_ext / g

    def arbitrage_step(self, s_ext: float) -> Tuple[float, float]:
        """Apply one arbitrage step for the retained-fee pool.

        When S* moves, the pool's internal price S = (c/X)^{1/beta} must stay
        in [gamma*S*, S*/gamma].  If the external price violates the band,
        arbitrage adjusts X such that S = boundary value, while retaining
        fee revenues in the numeraire buffer K = Y - f(X).

        After adjusting X to x_new = q(s_new), Y is updated so that
        the buffer K is preserved (retained fees never leave):
            y_new = f(x_new) + K_old

        This ensures K is non-decreasing (Section 2.2.3).

        Returns
        -------
        (dx, dy) : changes in risky and numeraire reserves.
        """
        lower, upper = self.no_arb_band(s_ext)
        s_curr = self.pool_price
        dx, dy = 0.0, 0.0

        if s_curr < lower:
            # External price rose; arbitrageur buys X, pool price rises to lower.
            s_new = lower
            x_new = inventory_schedule_quantity(s_new, self.c, self.beta)
            k_buf = self.fee_buffer
            y_new = inventory_schedule_f(x_new, self.c, self.beta) + k_buf
            dx, dy = x_new - self.x, y_new - self.y
            self.x, self.y = x_new, y_new

        elif s_curr > upper:
            # External price fell; arbitrageur sells X, pool price falls to upper.
            s_new = upper
            x_new = inventory_schedule_quantity(s_new, self.c, self.beta)
            k_buf = self.fee_buffer
            y_new = inventory_schedule_f(x_new, self.c, self.beta) + k_buf
            dx, dy = x_new - self.x, y_new - self.y
            self.x, self.y = x_new, y_new

        return dx, dy


@dataclass
class GMMPool:
    """Geometric Mean Market Maker pool for optimal investment boundaries.

    Invariant: X^k * Y = C  (Balancer-style weighted pool).
    Internal price: S = k * Y / X.
    Risky fraction: pi = S* X / (Y + S* X).

    The pool is designed to keep pi in [pi_minus, pi_plus] by choosing
    parameters k and tau from the target boundaries (Section 5.3):
        a = pi_minus/(1-pi_minus),  b = pi_plus/(1-pi_plus)
        k = sqrt(a * b),            gamma = sqrt(a / b)
        tau = 1 - gamma / (1 - lambda)
    """
    x: float       # risky reserve X
    y: float       # numeraire reserve Y
    c_inv: float   # GMM invariant C = X^k * Y
    k: float       # GMM weight parameter
    tau: float     # pool fee
    lam: float     # external TC lambda
    pi_minus: float  # lower no-trade boundary
    pi_plus: float   # upper no-trade boundary

    @property
    def gamma(self) -> float:
        return (1 - self.tau) * (1 - self.lam)

    @property
    def pool_price(self) -> float:
        return gmm_price(self.x, self.y, self.k)

    def risky_fraction(self, s_ext: float) -> float:
        return risky_fraction(s_ext, self.x, self.y)

    def no_arb_band(self, s_ext: float) -> Tuple[float, float]:
        g = self.gamma
        return g * s_ext, s_ext / g

    def liquidation_value(self, s_ext: float) -> float:
        return self.y + s_ext * (1 - self.lam) * self.x

    def arbitrage_step(self, s_ext: float) -> Tuple[float, float]:
        """Clamp pool price to [gamma*S*, S*/gamma] while maintaining GMM invariant.

        Returns (dx, dy).
        """
        lower, upper = self.no_arb_band(s_ext)
        s_curr = self.pool_price
        dx, dy = 0.0, 0.0

        if s_curr < lower:
            s_new = lower
            x_new, y_new = gmm_reserves_from_price(s_new, self.c_inv, self.k)
            dx, dy = x_new - self.x, y_new - self.y
            self.x, self.y = x_new, y_new

        elif s_curr > upper:
            s_new = upper
            x_new, y_new = gmm_reserves_from_price(s_new, self.c_inv, self.k)
            dx, dy = x_new - self.x, y_new - self.y
            self.x, self.y = x_new, y_new

        return dx, dy


# ---------------------------------------------------------------------------
# Parameter factory for optimal investment boundaries (Section 5.3)
# ---------------------------------------------------------------------------

def gmm_params_from_boundaries(
    pi_minus: float,
    pi_plus: float,
    lam: float,
) -> dict:
    """Derive GMM parameters from no-trade weight boundaries.

    Follows Section 5.3:
        a = pi_minus / (1 - pi_minus)
        b = pi_plus / (1 - pi_plus)
        k = sqrt(a * b)          (GMM weight parameter)
        gamma = sqrt(a / b)      (no-arb band half-width ratio)
        tau = 1 - gamma / (1 - lambda)   (pool fee)

    Parameters
    ----------
    pi_minus : lower no-trade boundary (fraction in risky)
    pi_plus  : upper no-trade boundary (fraction in risky)
    lam      : external proportional transaction cost lambda

    Returns
    -------
    dict with keys: a, b, k, gamma, tau
    """
    assert 0 < pi_minus < pi_plus < 1, "Need 0 < pi_minus < pi_plus < 1"
    a = pi_minus / (1 - pi_minus)
    b = pi_plus / (1 - pi_plus)
    k = np.sqrt(a * b)
    gamma = np.sqrt(a / b)
    tau = 1 - gamma / (1 - lam)
    assert tau >= 0, f"Infeasible: tau={tau:.4f} < 0.  gamma={gamma} > 1-lambda={1-lam}"
    return {"a": a, "b": b, "k": k, "gamma": gamma, "tau": tau}
