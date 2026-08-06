"""
Running Moskowitz, Ooi & Pedersen (2012) on China futures, via Ricequant.

replicate.py does this on global instruments and has AQR's own published factor
sitting next to it as a control. There is no such control here: nobody publishes
a China TSMOM factor, and the paper never looked at this market. So the honest
framing is different.

WHAT THIS FILE IS
    An out-of-sample test. The paper's sample is January 1985 to December 2009.
    Ricequant's continuous-contract service starts 2010-01-04. The two windows do
    not overlap by a single day, and the instruments are disjoint. Nothing here
    can "confirm" the paper's numbers; it can only ask whether the mechanism the
    paper describes shows up somewhere it was never fitted.

WHAT COUNTS AS SUCCESS
    Not a Sharpe ratio above 1. That would be a bonus, not the test. What we are
    checking is the paper's *signature*, and each of these can come out negative
    without anything being broken:

      1. Positions sized at 40%/vol really do realize ~40% vol   (the invariant)
      2. The diversified factor's vol lands far below 40%, because averaging
         many weakly-correlated bets is where the diversification shows up
      3. Most instruments individually profitable          (Fig. 2: 58/58)
      4. TSMOM beats a passive long at identical risk      (Fig. 3)

    Item 1 is the only one that is a bug if it fails. Items 2-4 are findings.

READ THE BREADTH TABLE BEFORE THE SHARPE RATIOS
    In 2010 this panel has 18 instruments and every one of them is a commodity;
    by 2026 it has 70 across three classes. An "all assets" factor computed over
    the early years is a commodity factor wearing a different label. MIN_BREADTH
    below drops months too thin to mean anything, and the per-era table exists so
    that the changing composition is visible rather than averaged away.
"""

import numpy as np
import pandas as pd

from data_rq import ASSET_CLASS, INSTRUMENT_NAME, futures_panel
from tsmom import (MONTHS_PER_YEAR, VOL_TARGET, build_tsmom, ex_ante_annual_vol,
                   performance)

# Months with fewer instruments than this are dropped before anything is
# reported. fig3_wind.py uses the same threshold for the same reason: the first
# months of a panel are a handful of contracts, and an equal-weighted average
# over three of them is noise with a confident-looking name.
MIN_BREADTH = 10

# The three classes present in China. The paper has a fourth, currencies, which
# has no liquid domestic futures market here.
CLASSES = ["commodity", "equity", "bond"]


def show(label, stats):
    """Print one row of performance statistics. Same shape as replicate.py."""
    print(f"  {label:<34}"
          f"Sharpe {stats['sharpe']:6.2f}   "
          f"ann_vol {stats['ann_vol']:6.2%}   "
          f"ann_mean {stats['ann_mean']:7.2%}   "
          f"maxDD {stats['max_drawdown']:7.2%}   "
          f"n={stats['months']}")


def sharpe_t_stats(per_inst):
    """Per-instrument Sharpe ratios with a proper t-statistic.

    replicate.py approximates a Sharpe ratio's standard error as 1/sqrt(years),
    which makes Sharpe*sqrt(years) roughly a t-statistic. That is fine for a
    headline. Here we want to make a claim about how many instruments are
    individually significant, so we run the real test instead: a one-sample
    t-test on the monthly returns, t = mean / (sd / sqrt(n)).

    The two agree closely -- the approximation is a first-order expansion of
    this -- but this one does not need a caveat attached to it.
    """
    rows = {}
    for col in per_inst.columns:
        r = per_inst[col].dropna()
        if len(r) < 24:                      # under two years, don't bother
            continue
        stats = performance(r)
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        rows[col] = {"sharpe": stats["sharpe"], "t": t, "months": len(r)}
    return pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)


def build(start=None, end=None):
    """The panel, the factor, and the passive benchmark, breadth-filtered.

    Returns (panel, div, per_inst, parts, passive) where `div` and `passive` have
    already had thin months removed. `per_inst` is left whole -- a single
    instrument's own strategy has no breadth problem.
    """
    panel = futures_panel(start, end)
    div, per_inst, parts = build_tsmom(panel)

    # The paper's Fig. 3 benchmark: the identical formula with sign() deleted,
    # i.e. always long, sized to the same 40% target. Same construction as
    # replicate.py and fig3_wind.py use.
    passive = ((VOL_TARGET / parts["vol_m"]).shift(1) * parts["monthly_ret"]).mean(axis=1)

    breadth = breadth_of(per_inst)
    keep = breadth[breadth >= MIN_BREADTH].index
    return panel, div.loc[keep].dropna(), per_inst, parts, passive.loc[keep].dropna()


def breadth_of(per_inst):
    """S_t -- how many instruments actually contribute to the average each month.

    Not parts["monthly_ret"].notna(): resample("ME").prod() over a month where an
    instrument has no data at all returns 1.0, the product of an empty set, so
    monthly_ret is 0.0 rather than NaN and counting notna() counts every column
    in every month, including years before the contract was listed. per_inst is
    the honest count -- it is NaN wherever the position was NaN, which is
    wherever there was no volatility estimate to size with.

    One residual case this does not catch: pandas' .ewm() carries its last value
    across NaN rows, so a contract that goes quiet for months keeps a stale
    volatility and therefore a non-NaN position, and books a fake 0% return. On
    this panel that is 13 cells out of 8,143, all of them FU (fuel oil, dormant
    from 2014 until its 2018 relaunch). Masking them changes the factor by 5e-5
    a month, so it is left alone rather than special-cased in the shared engine.
    """
    return per_inst.notna().sum(axis=1)


def main():
    print("=" * 104)
    print("TSMOM on China futures (Ricequant), 2010-2026 -- OUT OF SAMPLE vs the paper's 1985-2009")
    print("=" * 104)

    panel, div, per_inst, parts, passive = build()
    breadth = breadth_of(per_inst)
    print(f"  {panel.shape[1]} instruments   "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}   "
          f"{len(div)} usable months after the breadth filter "
          f"(>= {MIN_BREADTH} instruments)\n")

    # ------------------------------------------------------------------
    # The factor, whole and by asset class
    # ------------------------------------------------------------------
    show("TSMOM (all assets)", performance(div))
    for cls in CLASSES:
        cols = [c for c in panel.columns if ASSET_CLASS[c] == cls]
        series = per_inst[cols].mean(axis=1).loc[div.index].dropna()
        show(f"  TSMOM^{cls[:2].upper()} ({len(cols)})", performance(series))
    print("\n  paper, for reference: ~12% annualized vol, Sharpe > 1 (Section 4.1),")
    print("  on 58 instruments across FOUR asset classes. This panel has three.")

    # ------------------------------------------------------------------
    # The invariant. This is the one check that is a bug if it fails.
    # ------------------------------------------------------------------
    pos = parts["position"].to_numpy()
    assert np.isfinite(pos[~np.isnan(pos)]).all(), "non-finite position sizes"

    monthly = (1.0 + panel).resample("ME").prod() - 1.0
    size = VOL_TARGET / ex_ante_annual_vol(panel).resample("ME").last()
    realized = ((size.shift(1) * monthly).std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)).mean()
    print(f"\n  invariant -- mean realized vol per sized position: "
          f"{realized:.2f}  (target {VOL_TARGET:.2f})")

    # Diversification: each bet risks 40%, the portfolio should risk far less.
    print(f"  diversification -- 40% per position becomes "
          f"{performance(div)['ann_vol']:.1%} for the portfolio")

    # ------------------------------------------------------------------
    # Per-instrument -- the paper's Fig. 2
    # ------------------------------------------------------------------
    tab = sharpe_t_stats(per_inst)
    pos_n = int((tab["sharpe"] > 0).sum())
    sig_n = int(((tab["t"] > 1.96) & (tab["sharpe"] > 0)).sum())
    print(f"\n  per-instrument: {pos_n}/{len(tab)} positive Sharpe, "
          f"{sig_n} significant at 5% (two-sided t-test on monthly returns)")
    print(f"  paper Fig. 2:   58/58 positive, 52 significant")

    print("\n  best five:")
    for code, row in tab.head(5).iterrows():
        print(f"    {code:<4} {INSTRUMENT_NAME.get(code, code):<28} "
              f"Sharpe {row['sharpe']:+.2f}  t={row['t']:+.2f}  n={int(row['months'])}")
    print("  worst five:")
    for code, row in tab.tail(5).iterrows():
        print(f"    {code:<4} {INSTRUMENT_NAME.get(code, code):<28} "
              f"Sharpe {row['sharpe']:+.2f}  t={row['t']:+.2f}  n={int(row['months'])}")

    # ------------------------------------------------------------------
    # Does the signal earn its keep? -- the paper's Fig. 3 comparison
    # ------------------------------------------------------------------
    print("\n  TSMOM vs the same portfolio with sign() deleted (always long,")
    print("  identical 40%/vol sizing), over the identical months:")
    both = pd.concat([div.rename("ts"), passive.rename("pa")], axis=1).dropna()
    show("passive long (same sizing)", performance(both["pa"]))
    show("TSMOM", performance(both["ts"]))

    # ------------------------------------------------------------------
    # Composition changes over time, so show the eras rather than hide them
    # ------------------------------------------------------------------
    print("\n  by era (instrument count is the mean breadth in that window):")
    for lo, hi in [("2010", "2013"), ("2014", "2017"), ("2018", "2021"), ("2022", "2026")]:
        seg = div.loc[lo:hi]
        if len(seg) > 24:
            n = breadth.loc[lo:hi].mean()
            show(f"{lo}-{hi}  (~{n:.0f} instruments)", performance(seg))

    # ------------------------------------------------------------------
    # Cross-seam check: is the new data seam sane?
    # ------------------------------------------------------------------
    cross_seam_check()


def cross_seam_check():
    """Compare Ricequant and Wind on products that exist in both feeds.

    The cheapest available evidence that data_rq.py is not silently broken.
    Shanghai gold and COMEX gold are different contracts, quoted in different
    currencies, trading in different time zones -- but they track the same metal,
    so their daily returns should be strongly positively correlated. A near-zero
    correlation would mean the roll adjustment or the alignment is wrong.

    We do NOT expect these to be near 1.0: the CNY/USD move sits between them,
    the sessions only partly overlap, and Chinese contracts have daily price
    limits that the US ones do not.
    """
    print("\n" + "=" * 104)
    print("CROSS-SEAM CHECK -- Ricequant vs Wind on the same underlying commodity")
    print("=" * 104)

    try:
        from data_wind import futures_panel as wind_panel
        wind = wind_panel()
    except Exception as exc:
        print(f"  skipped: {exc}")
        return

    rq = futures_panel()
    pairs = [("AU", "GC.CMX", "Gold"), ("AG", "SI.CMX", "Silver"),
             ("CU", "HG.CMX", "Copper"), ("SC", "CL.NYM", "Crude oil"),
             ("M", "SM.CBT", "Soybean meal"), ("Y", "BO.CBT", "Soybean oil")]

    for rq_code, wind_code, label in pairs:
        if rq_code not in rq.columns or wind_code not in wind.columns:
            continue
        joined = pd.concat([rq[rq_code].rename("rq"), wind[wind_code].rename("wind")],
                           axis=1).dropna()
        if len(joined) < 250:
            continue
        corr = joined["rq"].corr(joined["wind"])
        flag = "  <-- suspicious" if corr < 0.15 else ""
        print(f"  {label:<14} {rq_code:>3} vs {wind_code:<9} "
              f"corr {corr:+.2f} over {len(joined)} days{flag}")

    print("\n  Expect moderate positive correlations. Near zero would mean the roll")
    print("  adjustment or the date alignment in data_rq.py is wrong.")


if __name__ == "__main__":
    main()
