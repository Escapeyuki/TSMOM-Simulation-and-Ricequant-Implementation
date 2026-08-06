"""
The paper's Table 3 on China futures: alpha, factor loadings, and the straddle.

Table 3 asks two separate questions about the diversified TSMOM factor.

  PANEL A -- Is the return explained by known factors? Regress TSMOM on the
      market and the standard style factors and look at the intercept. The paper
      gets an alpha of 1.58% per month against MSCI World + SMB/HML/UMD, and
      1.09% per month (t = 5.40) against the Asness-Moskowitz-Pedersen "value and
      momentum everywhere" factors. Both survive, which is the paper's claim that
      time series momentum is not repackaged cross-sectional momentum.

  PANEL C -- Does it look like an option straddle? Regress quarterly TSMOM on the
      market AND the market squared. A trend follower is long in rising markets
      and short in falling ones, so it should make money at both extremes: a U
      shape, which a positive coefficient on market-squared detects. The paper
      gets mkt +0.01 (t = 0.17) and mkt^2 +1.99 (t = 3.88) -- no net market
      exposure, strong convexity.

Panel B of the paper uses the AMP "everywhere" factors, which do not exist for
China in any form worth faking, so it is skipped rather than approximated twice.

THE REGRESSION CODE IS NOT NEW
    replicate.py::smile() already implements Panel C exactly, and replicate.ols()
    the rest. This file supplies China inputs to them. That is deliberate: if the
    straddle test disagrees between the global and China runs, the difference is
    in the data, because the code is character-for-character the same.

READ factors_rq.py BEFORE BELIEVING THE ALPHA
    Every control here is a substitute for something the paper used, and two of
    them are much stronger in China over this window than their US counterparts
    were over 1985-2009 -- the A-share size premium runs at a Sharpe of 1.6.
    Regressing on a high-Sharpe factor shrinks an intercept whether or not the
    exposure is real. The unconditional mean is printed next to the alpha for
    that reason.
"""

import numpy as np
import pandas as pd

from factors_rq import market_quarterly, monthly_factors
from replicate import ols, show_smile, smile
from replicate_rq import CLASSES, build
from data_rq import ASSET_CLASS
from tsmom import performance


def regress(label, series, factors, names=None):
    """One row of Panel A: alpha, its t-statistic, and the loadings."""
    names = names or list(factors.columns)
    joined = pd.concat([series.rename("r"), factors[names]], axis=1).dropna()
    table, r2 = ols(joined["r"], [joined[c] for c in names], names)

    alpha_m = table.loc["intercept", "coef"]
    print(f"  {label:<26} alpha {alpha_m:+.2%}/mo (t={table.loc['intercept', 't']:+.2f})   "
          f"R2 {r2:.2f}   n={len(joined)}")
    loadings = "   ".join(
        f"{n} {table.loc[n, 'coef']:+.2f} (t={table.loc[n, 't']:+.2f})" for n in names)
    print(f"  {'':<26} {loadings}")
    return table


def main():
    panel, div, per_inst, parts, passive = build()
    factors = monthly_factors()

    print("=" * 104)
    print("TABLE 3 PANEL A -- what explains the diversified TSMOM factor?")
    print("=" * 104)
    print(f"  China futures, {div.index[0].date()} -> {div.index[-1].date()}, "
          f"{len(div)} months\n")

    # The unconditional number first, so the alphas below have something to be
    # compared against rather than being read in isolation.
    stats = performance(div)
    t_mean = div.mean() / (div.std(ddof=1) / np.sqrt(len(div)))
    print(f"  before any controls:       mean {stats['ann_mean']:+.2%}/yr "
          f"({div.mean():+.2%}/mo, t={t_mean:+.2f}), vol {stats['ann_vol']:.2%}\n")

    # Paper Panel A: the market plus the three Fama-French style factors.
    regress("vs MKT + SMB/HML/UMD", div, factors, ["MKT", "SMB", "HML", "UMD"])
    print()
    # Paper Eq. 24: the same plus passive exposure to bonds and commodities.
    regress("vs all six (Eq. 24)", div, factors)
    print()
    # The control the paper cares most about in Fig. 3: is this just being long?
    # Not one of Table 3's own regressions, but it is the one exposure a China
    # commodity trend strategy is most likely to actually have.
    long_only = pd.concat([passive.rename("PASSIVE")], axis=1)
    regress("vs passive long", div, long_only, ["PASSIVE"])

    print(f"\n  paper Table 3 Panel A: alpha +1.58%/mo vs MSCI World + SMB/HML/UMD")

    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("PER ASSET CLASS -- alpha of each sleeve, paper Panels B-E")
    print("=" * 104)
    for cls in CLASSES:
        cols = [c for c in panel.columns if ASSET_CLASS[c] == cls]
        series = per_inst[cols].mean(axis=1).reindex(div.index).dropna()
        if len(series) < 36:
            continue
        regress(f"TSMOM^{cls[:2].upper()} ({len(cols)})", series, factors)
        print()

    # ------------------------------------------------------------------
    print("=" * 104)
    print("TABLE 3 PANEL C -- the straddle test")
    print("=" * 104)
    print("  Regress quarterly TSMOM on the market and the market squared.")
    print("  A trend follower should show no net market beta and positive convexity.\n")

    mkt_q = market_quarterly()
    show_smile("China TSMOM, 2010-2026", *smile(div, mkt_q))

    # The same test on the passive-long portfolio, as a control. A permanently
    # long book has a real market beta and no particular convexity, so this row
    # should look different from the one above. If it does not, the test is not
    # measuring what it claims to.
    show_smile("passive long (control)", *smile(passive, mkt_q))

    print("\n  paper Table 3 Panel C:         "
          "        mkt +0.01 (t=+0.17)   mkt^2 +1.99 (t=+3.88)")
    print("\n  The paper's market is MSCI World; here it is the CSI 300 TOTAL")
    print("  RETURN index in excess of the 3-month CGB yield -- the price index")
    print("  would understate it by the 2.2%/yr dividend yield. A China futures")
    print("  book has far less reason to")
    print("  co-move with A-shares than a global futures book has with world equity,")
    print("  so a flat market beta here is weaker evidence than the paper's.")


if __name__ == "__main__":
    main()
