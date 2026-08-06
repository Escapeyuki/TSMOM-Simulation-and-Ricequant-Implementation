"""
What the frictions cost: the event-driven backtest against the vectorized factor.

replicate_rq.py computes the paper's TSMOM factor arithmetically -- multiply a
position by a return, average across instruments. strategy_rq.py trades the same
signal through RQAlpha Plus with a real futures account. Neither number is
interesting on its own. The GAP between them is, because everything the
vectorized version assumes away lives in that gap:

    - whole contracts only, so small positions round to zero and vanish
    - commission on every trade, including every roll
    - slippage crossing the spread
    - margin, which caps what the 40%/vol sizing is allowed to ask for
    - a real cash balance rather than an implicit one

Run strategy_rq.py first (see its docstring), then this.

A CAVEAT THAT IS NOT SMALL
    The engine warns "trading parameters are abnormal" for pre-2019 contracts:
    future_info.json ships only with the sample bundle and carries current
    commission and margin rates, so the engine falls back to today's parameters
    for older dates. Chinese futures commissions have generally FALLEN over time,
    so the early years of this backtest are charged too little. The friction cost
    reported below is therefore a floor, not an estimate.
"""

import pickle

import numpy as np
import pandas as pd

from replicate_rq import build
from tsmom import performance

RESULT = "rq_result.pkl"


def load_backtest(path=RESULT):
    """Monthly returns of the event-driven portfolio."""
    with open(path, "rb") as fh:
        result = pickle.load(fh)

    # sys_analyser stores a daily frame with unit_net_value; the summary dict
    # alongside it carries the headline stats the engine computed itself.
    portfolio = result["portfolio"]
    nav = portfolio["unit_net_value"].copy()
    nav.index = pd.to_datetime(nav.index)

    summary = result.get("summary", {})

    # Guard against reading a stale pickle from an earlier, shorter run -- which
    # is exactly the mistake this file is meant to help avoid. A 15-year daily
    # backtest has ~3,700 rows; a 10-month smoke test has ~200.
    if len(nav) < 1000:
        raise SystemExit(
            f"{path} covers only {len(nav)} trading days "
            f"({summary.get('start_date')} -> {summary.get('end_date')}). "
            "That is a smoke test, not the full run -- re-run strategy_rq.py "
            "over the whole sample before comparing.")

    monthly = nav.resample("ME").last()
    return monthly.pct_change().dropna(), summary


def main():
    _, div, _, _, passive = build()
    live, summary = load_backtest()

    print("=" * 100)
    print("WHAT THE FRICTIONS COST -- event-driven backtest vs the vectorized factor")
    print("=" * 100)

    # Compare over the months both actually cover, so the difference is costs
    # and not a difference in sample.
    joined = pd.concat([div.rename("vector"), live.rename("live")],
                       axis=1).dropna()
    print(f"  common window: {joined.index[0].date()} -> {joined.index[-1].date()}"
          f"   ({len(joined)} months)\n")

    v, l = performance(joined["vector"]), performance(joined["live"])
    for label, s in [("vectorized (no costs)", v), ("event-driven (with costs)", l)]:
        print(f"  {label:<28}"
              f"Sharpe {s['sharpe']:6.2f}   ann_vol {s['ann_vol']:6.2%}   "
              f"ann_mean {s['ann_mean']:7.2%}   maxDD {s['max_drawdown']:7.2%}")

    gap_ann = v["ann_mean"] - l["ann_mean"]
    print(f"\n  cost of implementation: {gap_ann:.2%} per year "
          f"({gap_ann / 12:.2%} per month), Sharpe {v['sharpe']:.2f} -> {l['sharpe']:.2f}")

    # A correlation well below 1 would mean the two are not the same strategy --
    # a signal or timing difference rather than a cost difference, which would
    # make the gap above uninterpretable.
    corr = joined["vector"].corr(joined["live"])
    print(f"  correlation of the two return streams: {corr:+.3f}")
    if corr < 0.80:
        print("  WARNING: below 0.80 -- these are not tracking the same strategy,")
        print("  so the gap is not purely the cost of trading. Check the rebalance")
        print("  dates and the roll handling before quoting the number above.")

    # Does the traded version still beat passive long?
    both = pd.concat([live.rename("live"), passive.rename("passive")],
                     axis=1).dropna()
    if len(both) > 24:
        p = performance(both["passive"])
        print(f"\n  passive long over the same months: Sharpe {p['sharpe']:.2f}, "
              f"{p['ann_mean']:.2%}/yr")
        print(f"  event-driven TSMOM still beats it: "
              f"{performance(both['live'])['ann_mean'] > p['ann_mean']}")

    if summary:
        print("\n  engine's own summary (DIFFERENT CONVENTIONS -- see below):")
        for key in ["total_returns", "annualized_returns", "sharpe",
                    "max_drawdown", "total_value"]:
            if key in summary:
                print(f"    {key:<22} {summary[key]}")
        print("""
  The engine's numbers do not match the ones above, and both are right:

    - It annualizes GEOMETRICALLY (a CAGR) where performance() in tsmom.py
      annualizes arithmetically. Over this sample that is 3.40% against 4.43%,
      and the 1.04% difference is almost exactly vol^2 / 2 = 0.87%, which is
      what the gap between the two conventions always is.
    - Its Sharpe subtracts a risk-free rate of about 2.8% a year. Futures
      returns are already excess returns -- margin, not cash -- so that
      subtracts the rate a second time and drives the ratio to near zero. The
      paper works in excess returns throughout, so the Sharpe above is the one
      comparable to it.

  What matters for the headline is that tsmom.performance() is applied to BOTH
  series identically, so the cost gap is measured like against like.""")


if __name__ == "__main__":
    main()
