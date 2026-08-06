"""
Paper Figs. 2 and 3, rebuilt on China futures via Ricequant.

    Fig. 2  Annualized Sharpe ratio of the 12-month trend strategy, one bar per
            contract, grouped by asset class. The paper's version (p. 237) has
            every one of its 58 bars above zero -- that picture IS the paper's
            headline claim, which is why it is worth drawing the China version
            even when it comes out looking different.

    Fig. 3  Cumulative excess return of the diversified TSMOM portfolio against a
            diversified passive long position at identical risk, log scale
            (p. 239).

fig3_wind.py draws the same Fig. 3 on the Wind panel; its chart furniture --
log ticks, end-of-line dollar labels, the de-overlap rule, the footer -- is
reused here rather than reinvented, so the two figures can be laid side by side
and only the data differs.

WHY THE PASSIVE LINE IS THE RIGHT BENCHMARK
    It is TSMOM's own formula (Eq. 5) with sign() deleted: always +1 instead of
    +/-1, same 40%/vol sizing, same contracts, same rebalance dates. The only
    difference is whether the trend may flip a position short, so the gap between
    the lines is the value of the signal and nothing else.

A NOTE ON THE COLORS
    replicate.py's asset-class palette was checked before being reused and does
    not survive: its equity blue (#4281A4) and bond teal (#48A9A6) sit 11.9 apart
    in normal-vision OKLab dE, under the 15 floor, so the two would be hard to
    tell apart in adjacent bars -- and harder still for a colorblind reader. The
    palette below is re-stepped to clear all six checks. China has no currency
    sleeve, so only three hues are needed where replicate.py needed four.

Run: python fig_rq.py  ->  fig2_sharpe_by_instrument_rq.png
                          fig3_tsmom_vs_passive_rq.png
"""

import textwrap

import matplotlib
matplotlib.use("Agg")                 # write a file, do not open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from data_rq import ASSET_CLASS
from replicate_rq import CLASSES, MIN_BREADTH, build, sharpe_t_stats

FIG2 = "fig2_sharpe_by_instrument_rq.png"
FIG3 = "fig3_tsmom_vs_passive_rq.png"

INK = "#1A1A1A"        # TSMOM: the paper's heavy black line
GREY = "#9098A1"       # Passive long: the paper's thin grey line
AXIS = "#4D4D4D"

# Validated with the dataviz palette checker: all six checks pass in light mode
# (worst adjacent pair dE 25.9 normal / 18.2 protan, all >= 3:1 on contrast).
CLASS_COLOR = {"commodity": "#B4504F", "equity": "#0F72C4", "bond": "#B8891F"}


# ---- Fig. 2 ----------------------------------------------------------------

def draw_fig2(tab):
    """One bar per contract, sorted within asset class, colored by class."""
    tab = tab.copy()
    tab["cls"] = [ASSET_CLASS[c] for c in tab.index]
    # Group by class, then by Sharpe inside each class, so the eye can compare
    # classes as blocks and instruments within a block.
    order = pd.concat([tab[tab["cls"] == cls].sort_values("sharpe", ascending=False)
                       for cls in CLASSES])

    fig, ax = plt.subplots(figsize=(13, 5.4))
    x = np.arange(len(order))
    colors = [CLASS_COLOR[c] for c in order["cls"]]
    # A 2px surface gap between adjacent bars, so neighbours never merge into
    # one block of color.
    ax.bar(x, order["sharpe"], color=colors, width=0.74,
           linewidth=0.6, edgecolor="white")

    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(order.index, rotation=90, fontsize=7.5)
    ax.set_xlim(-1, len(order))
    ax.set_ylabel("Annualized Sharpe ratio", fontsize=10.5, color=AXIS, labelpad=8)

    # Legend is present because there are three series; the class blocks plus
    # this make identity readable without relying on color alone.
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c in CLASSES]
    labels = [f"{c} ({int((order['cls'] == c).sum())})" for c in CLASSES]
    ax.legend(handles, labels, fontsize=9.5, loc="upper right", frameon=False,
              ncol=3, borderaxespad=0.6)

    positive = int((order["sharpe"] > 0).sum())
    ax.set_title("Sharpe ratio of the 12-month time series momentum strategy, "
                 "by contract\n"
                 f"China futures, 2010–2026 — {positive} of {len(order)} positive "
                 "(paper Fig. 2: 58 of 58)",
                 fontsize=13, color=INK, pad=12)

    for side in ax.spines.values():
        side.set_color("#C7CCD1")
        side.set_linewidth(0.9)
    ax.tick_params(colors=AXIS, labelsize=9, length=3, width=0.8)
    ax.grid(True, axis="y", which="major", color="#E4E7EA", lw=0.7)
    ax.set_axisbelow(True)

    method = textwrap.fill(
        "Each contract's own strategy: go long if its excess return over the past "
        "12 months was positive, short if negative, sized to 40% annualized "
        "volatility, held one month (Moskowitz, Ooi & Pedersen 2012, Eq. 5). "
        "Sharpe ratios are gross of transaction costs, computed on the roll-adjusted "
        "dominant continuous contract from Ricequant. Contracts with under 24 months "
        "of strategy history are omitted.", width=150)
    fig.text(0.012, 0.005, method, va="top", fontsize=8, color="#6B7178",
             linespacing=1.45)

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(FIG2, dpi=160, bbox_inches="tight", facecolor="white")


# ---- Fig. 3 ----------------------------------------------------------------

def wealth_curves(div, passive):
    """Both series as growth of $100, joined on year-month.

    Join on the period, not the timestamp: resample("ME") stamps the last
    *calendar* day, and anything stamped with the last *trading* day would look
    like a different date and silently drop out of the join. Same trick, and same
    reason, as fig3_wind.py.
    """
    def by_month(s, label):
        return s.rename(label).set_axis(s.index.to_period("M"))

    curves = pd.concat([by_month(div, "Time Series Momentum"),
                        by_month(passive, "Passive Long (same risk)")],
                       axis=1).dropna()

    # Both start at $100 in the month before the sample, so neither gets a head
    # start from its own first return.
    curves.loc[curves.index[0] - 1] = 0.0
    curves = curves.sort_index()
    curves.index = curves.index.to_timestamp(how="end").normalize()
    return 100 * (1 + curves).cumprod()


def draw_fig3(wealth, stats, n_instruments):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    styles = [(wealth.columns[0], INK, 1.9), (wealth.columns[1], GREY, 1.5)]
    for name, color, width in styles:
        ax.plot(wealth.index, wealth[name], color=color, lw=width,
                label=name, solid_joinstyle="round")

    # Direct labels at the line ends, pushed apart only when close enough to
    # overlap on a log axis.
    ends = wealth.iloc[-1]
    apart = abs(np.log10(ends.iloc[0] / ends.iloc[1])) > 0.055
    for name, color, _ in styles:
        if apart:
            dy = -3.0
        else:
            dy = 6.0 if ends[name] == ends.max() else -13.0
        ax.annotate(f"${ends[name]:,.0f}", (wealth.index[-1], ends[name]),
                    xytext=(7, dy), textcoords="offset points",
                    fontsize=9.5, color=color, fontweight="bold")

    ax.set_yscale("log")
    top = wealth.to_numpy().max()
    bottom = wealth.to_numpy().min()
    # Half-decade ticks, and low ones too: the passive book falls to about $22 in
    # 2015, and without ticks down there that whole stretch of the chart has no
    # gridline to read a level against.
    ticks = [t for t in (10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000)
             if t <= top * 1.6 and t >= bottom * 0.6]
    ax.set_yticks(ticks)
    ax.set_ylim(min(90, bottom * 0.94), top * 1.7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    first, final = wealth.index[0].year, wealth.index[-1].year
    ax.set_title("Growth of $100: time series momentum versus a diversified\n"
                 f"passive long position, China futures, {first}–{final}",
                 fontsize=13.5, color=INK, pad=12)
    ax.set_xlabel("Date", fontsize=10.5, color=AXIS, labelpad=8)
    ax.set_ylabel("Growth of $100 (log scale)", fontsize=10.5, color=AXIS,
                  labelpad=8)

    years = pd.date_range(f"{first}-12-31", f"{final}-12-31", freq="2YE")
    ax.set_xticks(years)
    ax.set_xticklabels([d.year for d in years], rotation=90)
    ax.set_xlim(wealth.index[0], wealth.index[-1] + pd.Timedelta(days=300))

    ax.legend(fontsize=10, loc="upper left", frameon=False,
              handlelength=2.4, borderaxespad=0.9, labelspacing=0.6)

    for side in ax.spines.values():
        side.set_color("#C7CCD1")
        side.set_linewidth(0.9)
    ax.tick_params(colors=AXIS, labelsize=9, length=3, width=0.8)
    ax.grid(True, axis="y", which="major", color="#E4E7EA", lw=0.7)
    ax.set_axisbelow(True)

    summary = "  |  ".join(
        f"{name}: {s['ann_mean']:.1%} p.a., {s['ann_vol']:.1%} vol, "
        f"Sharpe {s['sharpe']:.2f}, max drawdown {s['max_drawdown']:.0%}"
        for name, s in stats.items())
    method = textwrap.fill(
        "TSMOM: 12-month look-back, 1-month hold, each position sized to 40% "
        f"annualized volatility, equal-weighted across the {n_instruments} China "
        "futures in data_rq.py (Moskowitz, Ooi & Pedersen 2012, Eq. 5). The passive "
        "long book holds the same contracts at the same sizes but is always long, so "
        "the gap between the lines is the trend signal alone. Futures returns are "
        "excess returns by construction. Months with fewer than "
        f"{MIN_BREADTH} contracts available are excluded. Gross of costs; see "
        "strategy_rq.py for the same strategy with commissions, slippage and margin.",
        width=132)

    fig.text(0.012, 0.005, f"{summary}\n\n{method}", va="top", fontsize=8,
             color="#6B7178", linespacing=1.45)

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(FIG3, dpi=160, bbox_inches="tight", facecolor="white")


def main():
    panel, div, per_inst, parts, passive = build()

    tab = sharpe_t_stats(per_inst)
    draw_fig2(tab)
    print(f"Saved -> {FIG2}   ({int((tab['sharpe'] > 0).sum())}/{len(tab)} positive)")

    wealth = wealth_curves(div, passive)
    from tsmom import performance
    stats = {name: performance(
        wealth[name].pct_change().dropna()) for name in wealth.columns}
    draw_fig3(wealth, stats, panel.shape[1])

    print(f"Saved -> {FIG3}")
    print(f"{wealth.index[1].date()} -> {wealth.index[-1].date()}  "
          f"({len(wealth) - 1} months)")
    for name, s in stats.items():
        print(f"  {name:<30} final ${wealth[name].iloc[-1]:>8,.0f}   "
              f"ann {s['ann_mean']:6.2%}   vol {s['ann_vol']:6.2%}   "
              f"Sharpe {s['sharpe']:5.2f}   maxDD {s['max_drawdown']:7.2%}")


if __name__ == "__main__":
    main()
