"""
Replicating Moskowitz, Ooi & Pedersen (2012) on real data.

Run this file. It prints two things and then cross-checks them.

TARGET A -- the published factor.
    AQR (one of the authors' firm) publishes the exact TSMOM return series the
    paper used, 1985-2009. There is no data risk at all in this file, so if our
    performance() function disagrees with the paper's reported numbers, the bug
    is in our maths and nowhere else. This is the control.

TARGET B -- our own rebuild.
    Feed ~39 free instruments (see data.py) through build_tsmom() and see what
    comes out. This runs 2000-present, not 1985-2009, on 39 instruments, not 58.

WHAT COUNTS AS SUCCESS
    Not matching the paper's Sharpe ratio to two decimals -- that would be
    impossible on a different instrument set over a different quarter-century,
    and if we did match it exactly, it would mean we had accidentally leaked
    the benchmark into the rebuild. What we want is the paper's *signature*:

      1. Sharpe ratio above 1                        (Section 4.1)
      2. About 12% annual volatility for the factor,
         even though each position risks 40%         (Section 4.1)
      3. Almost every instrument profitable           (Fig. 2: 58 out of 58)
      4. A straddle-shaped payoff vs the stock market (Fig. 4, Table 3 Panel C)
"""

import numpy as np
import pandas as pd

from data import ASSET_CLASS, INSTRUMENT_NAME, futures_panel, load_aqr_factors
from tsmom import (MONTHS_PER_YEAR, VOL_TARGET, build_tsmom, ex_ante_annual_vol,
                   performance)


def ols(y, X, names):
    """Ordinary least squares regression with t-statistics.

    Written out by hand rather than importing statsmodels, because we only need
    three regressions and this is about fifteen lines.

    y     -- what we are explaining (a return series)
    X     -- list of explanatory series
    names -- labels for those series, for the printed table

    A t-statistic above about 2 in absolute value is the usual "this is
    probably not noise" threshold at the 5% level.
    """
    y = np.asarray(y, float)

    # Stack a column of 1s (that gives us the intercept) next to the predictors.
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])

    # Least squares: find the coefficients that minimise squared error.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    # Residuals = what the model failed to explain.
    resid = y - X @ beta
    dof = len(y) - X.shape[1]           # degrees of freedom
    s2 = resid @ resid / dof            # estimated variance of the errors

    # Standard errors sit on the diagonal of s2 * (X'X)^-1.
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))

    # R-squared: the fraction of the variation in y the model explains.
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))

    table = pd.DataFrame({"coef": beta, "t": beta / se},
                         index=["intercept"] + list(names))
    return table, r2


def smile(tsmom_monthly, mkt_quarterly):
    """The straddle test -- paper Table 3, Panel C.

    Regress quarterly TSMOM returns on the market return AND the market return
    squared:

        tsmom = a + b1 * market + b2 * market^2

    Why squared? A trend follower ends up long in rising markets and short in
    falling ones, so it makes money at *both* extremes -- like owning an option
    straddle. That is a U shape, and a U shape is what a positive coefficient on
    market^2 detects. Meanwhile b1, the plain market beta, should be near zero:
    TSMOM is not secretly just a long stock position.

    The paper gets market +0.01 (t = 0.17) and market^2 +1.99 (t = 3.88).

    Quarterly, not monthly, because the paper uses non-overlapping quarterly
    returns here to avoid markets in different time zones closing at different
    moments and blurring the relationship.
    """
    # Compound the monthly TSMOM returns up to quarterly. "QE" = quarter end.
    ts_q = (1 + tsmom_monthly).resample("QE").prod() - 1

    # Line the two series up by date and drop quarters missing either one.
    q = pd.concat([ts_q.rename("ts"), mkt_quarterly.rename("mkt")],
                  axis=1, sort=True).dropna()

    table, _ = ols(q["ts"], [q["mkt"], q["mkt"] ** 2], ["mkt", "mkt^2"])
    return table, len(q)


def show_smile(label, tab, n):
    """Print one row of the straddle test."""
    print(f"  {label:<30} n={n:3}   "
          f"mkt {tab.loc['mkt', 'coef']:+.2f} (t={tab.loc['mkt', 't']:+.2f})   "
          f"mkt^2 {tab.loc['mkt^2', 'coef']:+.2f} (t={tab.loc['mkt^2', 't']:+.2f})")


def show(label, stats):
    """Print one row of performance statistics."""
    print(f"  {label:<28}"
          f"Sharpe {stats['sharpe']:6.2f}   "
          f"ann_vol {stats['ann_vol']:6.2%}   "
          f"ann_mean {stats['ann_mean']:7.2%}   "
          f"maxDD {stats['max_drawdown']:7.2%}   "
          f"n={stats['months']}")


def main():
    # =====================================================================
    # TARGET A -- the paper's own published factor
    # =====================================================================
    print("=" * 100)
    print("TARGET A -- AQR's published factor (the paper's own output, 1985-2009)")
    print("=" * 100)

    aqr = load_aqr_factors("paper")
    aqr_tsmom = aqr["TSMOM"].dropna()
    show("TSMOM (all assets)", performance(aqr_tsmom))

    # The same file also breaks the factor down by asset class, the way the
    # paper reports TSMOM^COM, TSMOM^EQ, TSMOM^FI and TSMOM^FX.
    for col in [c for c in aqr.columns if c != "TSMOM"]:
        show(f"  {col}", performance(aqr[col].dropna()))
    print(f"\n  paper says: ~12% annualized vol, Sharpe > 1  (Section 4.1)")

    # =====================================================================
    # TARGET B -- our rebuild from free data
    # =====================================================================
    print("\n" + "=" * 100)
    print("TARGET B -- rebuilt from free instruments, 2000-present")
    print("=" * 100)

    panel = futures_panel()                        # daily excess returns
    div, per_inst, parts = build_tsmom(panel)      # the engine from tsmom.py
    print(f"  {panel.shape[1]} instruments   "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}   "
          f"{int(div.notna().sum())} usable months\n")

    show("TSMOM (all assets)", performance(div))

    # Same construction, one asset class at a time (paper Section 4.1, last
    # line: "We also consider TSMOM strategies by asset class constructed
    # analogously"). Averaging a subset of the columns is all it takes.
    for cls in ["commodity", "equity", "bond", "currency"]:
        cols = [c for c in panel.columns if ASSET_CLASS[c] == cls]
        show(f"  TSMOM^{cls[:2].upper()} ({len(cols)})",
             performance(per_inst[cols].mean(axis=1)))

    # Sanity check: dividing 40% by a volatility of zero would give infinity,
    # which would silently poison every average downstream. The min_periods
    # argument in ex_ante_annual_vol() prevents it; this proves it worked.
    pos = parts["position"].to_numpy()
    assert np.isfinite(pos[~np.isnan(pos)]).all(), "non-finite position sizes"

    # The sizing rule, checked on real data: a position sized at 40% / vol
    # should really show about 40% annualized volatility.
    monthly = (1.0 + panel).resample("ME").prod() - 1.0
    size = VOL_TARGET / ex_ante_annual_vol(panel).resample("ME").last()
    realized = ((size.shift(1) * monthly).std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)).mean()
    print(f"\n  invariant -- mean realized vol per sized position: "
          f"{realized:.2f}  (target {VOL_TARGET:.2f})")

    # Per-instrument Sharpe ratios -- our version of the paper's Fig. 2, which
    # shows all 58 contracts positive and 52 of them statistically significant
    # at the 5% level.
    sharpes = per_inst.apply(lambda col: performance(col)["sharpe"]).dropna()
    n_obs = per_inst.notna().sum()

    # A Sharpe ratio's own standard error is roughly 1/sqrt(years of data), so
    # Sharpe * sqrt(years) is approximately a t-statistic. 1.96 is the 5% cutoff.
    years = n_obs[sharpes.index] / MONTHS_PER_YEAR
    signif = (sharpes.abs() * np.sqrt(years) > 1.96)
    print(f"  per-instrument Sharpe: {int((sharpes > 0).sum())}/{len(sharpes)} positive, "
          f"{int((signif & (sharpes > 0)).sum())} significant at 5%   "
          f"(paper: 58/58, 52 significant)")
    print(f"  worst: {sharpes.idxmin()} {sharpes.min():.2f}   "
          f"best: {sharpes.idxmax()} {sharpes.max():.2f}")

    # =====================================================================
    # CROSS-CHECKS
    # =====================================================================
    print("\n" + "=" * 100)
    print("CROSS-CHECKS")
    print("=" * 100)

    # --- Check 1: does our rebuild move with AQR's own maintained factor? ---
    # Different instruments and a different sample, so we expect 0.6 to 0.8.
    # Around 0.3 would mean something is broken; around 1.0 would mean the
    # benchmark had somehow leaked into our rebuild.
    updated = load_aqr_factors("updated")["TSMOM"].dropna()
    joined = pd.concat([div.rename("rebuilt"), updated.rename("aqr")],
                       axis=1, sort=True).dropna()
    corr = joined["rebuilt"].corr(joined["aqr"])
    print(f"  rebuilt vs AQR's own updated factor: corr {corr:+.2f} over "
          f"{len(joined)} months "
          f"({joined.index[0].date()} -> {joined.index[-1].date()})\n")
    show("AQR updated, same window", performance(joined["aqr"]))
    show("rebuilt, same window", performance(joined["rebuilt"]))

    # --- Check 2: the straddle smile (Fig. 4 and Table 3 Panel C). ---
    # Note the order of the four rows below. We run the identical regression on
    # the paper's own factor over the paper's own window FIRST. If that row
    # reproduces the paper, the regression is right, and any difference in the
    # later rows is a real change in the world rather than a bug in this file.
    gspc = _spx_quarterly()
    print("\n  straddle smile -- quarterly TSMOM on market and market^2:")
    show_smile("AQR paper factor, 1985-2009",
               *smile(load_aqr_factors("paper")["TSMOM"].dropna(), gspc))
    show_smile("AQR updated, 1985-2009",
               *smile(updated.loc["1985":"2009"], gspc))
    show_smile("AQR updated, 2001-2026", *smile(updated.loc["2001":], gspc))
    show_smile("rebuilt, 2001-2026", *smile(div, gspc))
    print("  paper Table 3 Panel C:         "
          "        mkt +0.01 (t=+0.17)   mkt^2 +1.99 (t=+3.88)")

    # --- Check 3: is the trend signal actually adding anything? ---
    # The paper's Fig. 3 compares TSMOM to simply holding everything long at
    # the same risk. That is TSMOM's formula with the sign() removed, i.e.
    # always +1, so the code below is the position sizing with no signal.
    # TSMOM should beat it.
    passive = ((VOL_TARGET / parts["vol_m"]).shift(1) * parts["monthly_ret"]).mean(axis=1)
    print()
    show("passive long (same sizing)", performance(passive.dropna()))
    show("TSMOM", performance(div))

    # --- Check 4: has the effect decayed since publication? ---
    # Split AQR's own factor into eras. The instruments and the construction are
    # held fixed, so any difference across eras is the market, not the method.
    # Worth watching: the paper's sample ends in 2009 and it was published in
    # 2012, so most of our rebuild's window is *after* the anomaly was made
    # public -- and published anomalies have a habit of shrinking.
    print("\n  AQR's own factor by era (same instruments, same construction):")
    for lo, hi in [("1985", "1994"), ("1995", "2004"), ("2005", "2011"),
                   ("2012", "2018"), ("2019", "2026")]:
        seg = updated.loc[lo:hi]
        if len(seg) > 24:          # need at least two years to say anything
            show(f"{lo}-{hi}", performance(seg))

    _plot(div, passive, sharpes)


def _spx_quarterly():
    """S&P 500 quarterly returns back to 1984 -- the 'market' in the smile test.

    The paper uses the MSCI World Index; the S&P 500 is the free stand-in.
    """
    def go():
        import yfinance as yf
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            px = yf.download("^GSPC", start="1984-01-01", progress=False,
                             auto_adjust=True)["Close"]
        return px

    from data import _cached
    px = _cached("gspc", go).iloc[:, 0].dropna()
    return (1 + px.pct_change().dropna()).resample("QE").prod() - 1


def _plot(div, passive, sharpes):
    """Two charts: our Fig. 3 (equity curve) and our Fig. 2 (Sharpe by instrument)."""
    try:
        import matplotlib
        matplotlib.use("Agg")      # write a file instead of opening a window
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"\n(plots skipped: {e})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # LEFT -- growth of $1, TSMOM vs passive long, on a log scale like Fig. 3.
    cum = pd.concat([(1 + div.dropna()).cumprod().rename("TSMOM"),
                     (1 + passive.dropna()).cumprod().rename("Passive long")], axis=1)
    cum.plot(ax=axes[0], logy=True, title="Cumulative excess return (log scale)")
    axes[0].set_xlabel("")

    # RIGHT -- Sharpe ratio per instrument, sorted, coloured by asset class.
    # This is the paper's Fig. 2, where every bar is above zero.
    class_colors = {"commodity": "#C1666B", "equity": "#4281A4",
                    "bond": "#48A9A6", "currency": "#D4B483"}
    sorted_sharpes = sharpes.sort_values()

    # Colours are looked up while the index is still tickers...
    colors = [class_colors[ASSET_CLASS[ticker]] for ticker in sorted_sharpes.index]

    # ...then swap the tickers for readable names on the axis labels. .rename()
    # keeps the order, and leaves any ticker missing from the map untouched.
    sorted_sharpes.rename(INSTRUMENT_NAME).plot.bar(
        ax=axes[1], color=colors, width=0.8,
        title="Sharpe ratio by instrument")
    axes[1].axhline(0, color="black", lw=0.8)   # the zero line to read bars against
    axes[1].tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig("tsmom_replication.png", dpi=120)
    print("\nSaved -> tsmom_replication.png")


if __name__ == "__main__":
    main()
