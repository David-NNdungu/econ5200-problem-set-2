"""Reproducible experiment portfolios and deployable rules for ECON 5200 PS2.

Effects are standardized annual revenue units per deployment; one unit means
$1,000. Truth is retained for simulation audits and never used by apply_rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Rule:
    """A positive-direction p-value threshold, optionally with a design-power floor."""
    kind: Literal["threshold", "power_floor"] = "threshold"
    alpha: float = 0.05
    min_power: float = 0.80


def simulate_portfolio(
    n: int = 200, seed: int = 42, *, null_share: float = 0.80,
    positive_low: float = 0.40, positive_high: float = 1.20,
    n_min: int = 100, n_max: int = 5000, sigma: float = 10.0,
    design_effect: float = 0.80, alpha: float = 0.05,
) -> pd.DataFrame:
    """Simulate independent balanced A/B tests with known Gaussian outcome SD.

    Prior: point mass at zero with probability null_share, otherwise a positive
    Uniform(positive_low, positive_high). Sample sizes per arm are floor of a
    log-uniform draw on [n_min, n_max+1), capped at n_max. With n observations
    in EACH arm, SE = sigma*sqrt(2/n). Use a pre-specified one-sided z test.
    design_power uses a common, observable planning effect; oracle_power uses
    truth and is for diagnosis ONLY. Seeds are fixed before inspecting results.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if (not 0 <= null_share <= 1 or not 0 < positive_low < positive_high
            or not 2 <= n_min <= n_max or sigma <= 0 or design_effect <= 0
            or not 0 < alpha < 1):
        raise ValueError("invalid simulation parameters")
    rng = np.random.default_rng(seed)
    is_null = rng.random(n) < null_share
    positive_draw = rng.uniform(positive_low, positive_high, n)
    truth = np.where(is_null, 0.0, positive_draw)
    n_per_arm = np.minimum(
        np.floor(np.exp(rng.uniform(np.log(n_min), np.log(n_max + 1), n))).astype(int),
        n_max,
    )
    se = sigma * np.sqrt(2.0 / n_per_arm)
    estimate = rng.normal(truth, se)
    critical_z = stats.norm.isf(alpha)
    return pd.DataFrame({
        "experiment_id": np.arange(1, n + 1), "true_effect": truth,
        "is_null": is_null, "n_per_arm": n_per_arm,
        "n_total": 2 * n_per_arm, "estimate": estimate, "se": se,
        "z": estimate / se, "p_value": stats.norm.sf(estimate / se),
        "design_effect": np.full(n, design_effect),
        "design_power": stats.norm.sf(critical_z - design_effect / se),
        "oracle_power": stats.norm.sf(critical_z - truth / se),
    })


def apply_rule(log: pd.DataFrame, rule: Rule) -> pd.Series:
    """Return Boolean ship decisions without consulting true_effect or is_null.

    threshold: p_value < alpha. power_floor: the same AND design_power >=
    min_power. Recomputes planning power at rule.alpha using design_effect,
    never observed or true effect estimates.
    """
    if not 0 < rule.alpha < 1 or not 0 <= rule.min_power <= 1:
        raise ValueError("invalid rule parameters")
    ship = log["p_value"] < rule.alpha
    if rule.kind == "power_floor":
        power = stats.norm.sf(stats.norm.isf(rule.alpha) - log["design_effect"] / log["se"])
        ship = ship & (power >= rule.min_power)
    elif rule.kind != "threshold":
        raise ValueError(f"unknown rule kind: {rule.kind}")
    return ship.astype(bool).rename("ship")


def evaluate(log: pd.DataFrame, rule: Rule) -> dict[str, float | int]:
    """Audit realised return, ship count, and false ships / all ships.

    In this prior the null is exactly zero. No-shipment false-ship rate is NaN,
    not zero, because the denominator is empty. Returns are sums, not averages.
    """
    ship = apply_rule(log, rule)
    count = int(ship.sum())
    false = int((ship & log["is_null"]).sum())
    return {"realised_return": float(log.loc[ship, "true_effect"].sum()),
            "ship_count": count, "false_ship_count": false,
            "false_ship_rate": false / count if count else float("nan")}


if __name__ == "__main__":
    portfolio = simulate_portfolio(200, 42)
    client = Rule()
    alternative = Rule("power_floor", min_power=0.80)
    assert len(portfolio) == 200 and portfolio.experiment_id.is_unique
    assert portfolio.n_per_arm.between(100, 5000).all()
    assert np.allclose(portfolio.se, 10 * np.sqrt(2 / portfolio.n_per_arm))
    pd.testing.assert_frame_equal(portfolio, simulate_portfolio(200, 42))
    assert (apply_rule(portfolio, alternative) <= apply_rule(portfolio, client)).all()
    blind = portfolio.drop(columns=["true_effect", "is_null", "oracle_power"])
    pd.testing.assert_series_equal(apply_rule(blind, alternative), apply_rule(portfolio, alternative))
    for label, rule in [("client", client), ("alternative", alternative)]:
        print(label, evaluate(portfolio, rule))
    print("Six assertions passed.")
