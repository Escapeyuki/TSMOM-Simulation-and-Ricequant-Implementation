"""
Paper Fig. 3, rebuilt on the Wind data: TSMOM vs a diversified passive long.

The paper's Fig. 3 (p. 239) plots the cumulative excess return of the
diversified time series momentum portfolio against "a diversified portfolio of
the possible long position in every futures contract we study", on a log scale,
January 1985 to December 2009. This is the same chart, with one change: both
lines are built here, from the 42 contracts in prices_wind.csv, through the
engine in tsmom.py. Nothing on the chart is copied from the paper's output.

WHAT MAKES THIS THE RIGHT BENCHMARK
    The passive line is TSMOM's own formula (Eq. 5) with sign() deleted -- the
    position is always +1 instead of +/-1, and the 40%/vol sizing is untouched:

        r_passive[s, t+1] = (40% / vol[s, t]) * r[s, t+1],  averaged over s

    So the two portfolios hold the same contracts, in the same sizes, with the
    same ex ante volatility, rebalanced on the same schedule. The *only* thing
    that differs is whether the 12-month trend is allowed to flip a position
    short. The gap between the lines is therefore the value of the signal and
    nothing else -- which is exactly what a benchmark should isolate, and the
    reason the paper chose this comparison over an equity index.

BOTH LINES ARE EXCESS RETURNS, above the 3-month T-bill. Futures need margin,
not cash, so they are excess returns by construction; data_wind.py subtracts the
T-bill from the cash indexes so they belong on the same axis.

Run: python fig3_wind.py  ->  fig3_tsmom_vs_passive_wind.png
"""

import textwrap

import matplotlib
matplotlib.use("Agg")                 # write a file, do not open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from data_wind import futures_panel
from tsmom import VOL_TARGET, build_tsmom, performance

# A month in which only a handful of contracts have a 12-month signal is not a
# diversified portfolio; it is one or two bets. The first such months are cut so
# the curve does not start on noise. (Jan 1992 has 3 instruments; Feb has 16.)
MIN_INSTRUMENTS = 10

OUT = "fig3_tsmom_vs_passive_wind.png"

INK = "#1A1A1A"        # TSMOM: the paper's heavy black line
GREY = "#9098A1"       # Passive long: the paper's thin grey line
AXIS = "#4D4D4D"


def build():
    """Return (wealth curve, stats) -- both series indexed by month end."""
    panel = futures_panel()
    tsmom, per_instrument, parts = build_tsmom(panel)

    # The passive long book: build_tsmom's position sizing with the signal left
    # out. shift(1) for the same reason it is there in tsmom.py -- the size held
    # during month t is set by the volatility known at the end of month t-1.
    passive = ((VOL_TARGET / parts["vol_m"]).shift(1)
               * parts["monthly_ret"]).mean(axis=1)

    # Both lines are restricted to the months TSMOM is diversified enough to
    # exist, so the comparison runs over one identical window.
    breadth = per_instrument.notna().sum(axis=1)
    keep = breadth >= MIN_INSTRUMENTS
    tsmom = tsmom[keep].dropna()
    passive = passive[keep].dropna()

    # Stop at the last month the data covers *completely*, so a part-finished
    # month cannot show up as a divergence at the right-hand edge, which is
    # exactly where the eye goes. Whether the last month is whole is a property
    # of the file, so it is tested rather than hard-coded.
    cut = panel.index[-1]
    last = cut.to_period("M")
    if cut != cut + pd.offsets.MonthEnd(0):
        last -= 1

    # Join on year-month, not on the timestamp. resample("ME") stamps the last
    # *calendar* day; any series stamped with the last *trading* day would look
    # like a different date and silently drop out of the join.
    def by_month(s, label):
        return s.rename(label).set_axis(s.index.to_period("M"))

    curves = pd.concat([by_month(tsmom, "Time Series Momentum"),
                        by_month(passive, "Passive Long (same risk)")],
                       axis=1).dropna()
    curves = curves.loc[:last]

    stats = {name: performance(curves[name]) for name in curves.columns}

    # Both start at $100 in the month before the sample, so neither gets a head
    # start from its own first return.
    curves.loc[curves.index[0] - 1] = 0.0
    curves = curves.sort_index()
    curves.index = curves.index.to_timestamp(how="end").normalize()

    return 100 * (1 + curves).cumprod(), stats


def draw(wealth, stats):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    styles = [(wealth.columns[0], INK, 1.9), (wealth.columns[1], GREY, 1.5)]
    for name, color, width in styles:
        ax.plot(wealth.index, wealth[name], color=color, lw=width,
                label=name, solid_joinstyle="round")

    # Direct labels at the line ends, so identity survives without the legend.
    # The two lines finish within a few percent of each other, which on a log
    # axis is a few pixels, so the labels are pushed apart when they are close
    # enough to overlap -- and left alone when they are not.
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

    # The paper's sample spans three decades of wealth and so labels only
    # $100 / $1,000 / $10,000 / $100,000. This one spans about 1.5 decades, and
    # two labelled ticks is not enough to read a level off, so the half-decades
    # are labelled too. Ticks are picked to bracket the data either way.
    ax.set_yscale("log")
    top = wealth.to_numpy().max()
    ticks = [t for t in (100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000,
                         50_000, 100_000) if t <= top * 1.6]
    ax.set_yticks(ticks)
    ax.set_ylim(min(90, wealth.to_numpy().min() * 0.94), top * 1.7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    first, final = wealth.index[0].year, wealth.index[-1].year
    ax.set_title("Growth of $100: time series momentum versus a diversified\n"
                 f"passive long position, cumulative excess returns, {first}–{final}",
                 fontsize=13.5, color=INK, pad=12)
    ax.set_xlabel("Date", fontsize=10.5, color=AXIS, labelpad=8)
    ax.set_ylabel("Growth of $100 (log scale)", fontsize=10.5, color=AXIS,
                  labelpad=8)

    # The paper labels every other year, turned on their side.
    years = pd.date_range(f"{first}-12-31", f"{final}-12-31", freq="2YE")
    ax.set_xticks(years)
    ax.set_xticklabels([d.year for d in years], rotation=90)
    ax.set_xlim(wealth.index[0], wealth.index[-1] + pd.Timedelta(days=400))

    # Stacked, not side by side: a two-column legend runs far enough to the
    # right to sit on top of the momentum line after 2005.
    ax.legend(fontsize=10, loc="upper left", frameon=False,
              handlelength=2.4, borderaxespad=0.9, labelspacing=0.6)

    # Recessive frame: keep the paper's boxed look, but let the data carry the
    # contrast rather than the furniture.
    for side in ax.spines.values():
        side.set_color("#C7CCD1")
        side.set_linewidth(0.9)
    ax.tick_params(colors=AXIS, labelsize=9, length=3, width=0.8)
    ax.grid(True, axis="y", which="major", color="#E4E7EA", lw=0.7)
    ax.set_axisbelow(True)

    # One text block, wrapped here rather than by matplotlib's wrap=True: that
    # rewraps at draw time, after bbox_inches="tight" has already resized the
    # canvas, and the method note lands on top of the statistics line.
    summary = "  |  ".join(
        f"{name}: {s['ann_mean']:.1%} p.a., {s['ann_vol']:.1%} vol, "
        f"Sharpe {s['sharpe']:.2f}, max drawdown {s['max_drawdown']:.0%}"
        for name, s in stats.items())
    method = textwrap.fill(
        "TSMOM: 12-month look-back, 1-month hold, each position sized to 40% "
        "annualized volatility, equal-weighted across the 42 futures, forwards "
        "and cash indexes in prices_wind.csv (Moskowitz, Ooi & Pedersen 2012, "
        "Eq. 5). The passive long book holds the same contracts at the same "
        "sizes but is always long, so the gap between the lines is the trend "
        "signal alone. Both are net of the 3-month T-bill.", width=132)

    fig.text(0.012, 0.005, f"{summary}\n\n{method}", va="top", fontsize=8,
             color="#6B7178", linespacing=1.45)

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")


def main():
    wealth, stats = build()
    draw(wealth, stats)

    print(f"{wealth.index[1].date()} -> {wealth.index[-1].date()}  "
          f"({len(wealth) - 1} months)")
    for name, s in stats.items():
        print(f"  {name:<36} final ${wealth[name].iloc[-1]:>9,.0f}   "
              f"ann {s['ann_mean']:6.2%}   vol {s['ann_vol']:6.2%}   "
              f"Sharpe {s['sharpe']:5.2f}   maxDD {s['max_drawdown']:7.2%}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
