"""
Fig. 5 -- China against the US against AQR's own factor, one asset class at a time.

WHY THIS FIGURE EXISTS
    README_RQ.md's conclusion is that the paper's effect does not replicate on
    China futures: a diversified Sharpe of 0.62 against the paper's >1. That
    comparison has a hole in it. The paper's number is 1985-2009 and China's data
    starts in 2010, so "China is worse" and "the last fifteen years were worse"
    are the same observation wearing two different labels, and nothing in the
    repo could tell them apart.

    AQR publishes the authors' own TSMOM factor, monthly, from 1985 to today --
    it is already in the repo, read by replicate.py as its control. Running it
    forward past the paper's sample turns it into the era control this comparison
    needed: the same construction, by the same people, on the same global
    instruments, measured over China's window instead of the paper's.

    Sharpe 1.41 in the paper's sample. 0.40 since. The decay is not a China
    result.

WHAT IS PLOTTED
    Growth of $100 in each asset-class sleeve, over the months every series in
    that panel is live, log scale. Four series: China, the two US data feeds, and
    AQR's published sleeve for that class. The US appears twice on purpose --
    Yahoo and Wind are two vendors quoting the same contracts, so the gap between
    the solid and dashed blue lines is measurement noise, and it is the yardstick
    for reading the gap to the red line.

COLOR
    Hue is the market (red China, blue United States, gold AQR); line style is
    the data feed within a market. Three hues, validated on a white surface
    (fig_rq.py's palette, worst all-pairs dE 16.2 normal / 12.0 deutan).

Run: python fig5_class_sleeves.py  ->  fig5_class_sleeves_rq.png
"""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from compare_markets import MIN_MONTHS, class_table, sharpe

OUT = "fig5_class_sleeves_rq.png"

INK = "#1A1A1A"
AXIS = "#4D4D4D"
MUTED = "#6B7178"
RULE = "#C7CCD1"
GRID = "#E4E7EA"

# (source key, label, color, linewidth, dash). Hue is the market -- red China,
# blue United States -- and dash is the data feed within a market.
#
# AQR gets graphite rather than a third hue, for two reasons. It is the
# reference series, the authors' own answer, and a benchmark reading as
# recessive against the two things being compared is the right hierarchy. And a
# third hue does not survive the check: gold beside this red measures OKLab dE
# 5.7-8.2 under deuteranopia at every lightness that also clears contrast on a
# dark ground, so the dashboard could not carry it into dark mode. Two hues plus
# an ink line is colorblind-safe by construction in both.
SERIES = [
    ("cn", "China (Ricequant)", "#B4504F", 2.0, None),
    ("yahoo", "United States (Yahoo)", "#0F72C4", 1.8, None),
    ("wind", "United States (Wind)", "#0F72C4", 1.3, (0, (5, 2.6))),
    ("aqr", "AQR published factor", "#55565B", 1.7, (0, (1.6, 1.8))),
]

PANELS = [("all", "All assets"), ("commodity", "Commodity futures"),
          ("equity", "Equity index futures"), ("bond", "Government bond futures")]


def wealth(members):
    """Growth of $100 for every member of one panel, on their shared months.

    Seeded with a zero-return row one month before the sample so no line gets a
    head start from its own first return -- the same construction fig_rq.py and
    fig3_wind.py use, and the same reason for joining on periods rather than
    timestamps.
    """
    frame = pd.concat(members, axis=1).dropna()
    if len(frame) < MIN_MONTHS:
        return None, None
    sharpes = {name: sharpe(frame[name]) for name in frame.columns}
    frame.loc[frame.index[0] - 1] = 0.0
    frame = frame.sort_index()
    frame.index = frame.index.to_timestamp(how="end").normalize()
    return 100 * (1 + frame).cumprod(), sharpes


def draw_panel(ax, curves, sharpes, title):
    for key, label, color, lw, dash in SERIES:
        if key not in curves.columns:
            continue
        ax.plot(curves.index, curves[key], color=color, lw=lw, label=label,
                linestyle=dash if dash else "-", solid_joinstyle="round",
                dash_capstyle="round")

    ax.set_yscale("log")
    top, bottom = curves.to_numpy().max(), curves.to_numpy().min()
    ticks = [t for t in (20, 50, 100, 200, 500, 1_000, 2_000)
             if bottom * 0.65 <= t <= top * 1.5]
    ax.set_yticks(ticks)
    ax.set_ylim(bottom * 0.9, top * 1.35)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    first, final = curves.index[0].year, curves.index[-1].year
    years = pd.date_range(f"{first}-12-31", f"{final}-12-31", freq="3YE")
    ax.set_xticks(years)
    ax.set_xticklabels([d.year for d in years])
    ax.set_xlim(curves.index[0], curves.index[-1])

    ax.set_title(f"{title}   ({len(curves) - 1} shared months)",
                 fontsize=11, color=INK, pad=8, loc="left")

    # A compact stats block instead of end-of-line labels: with four lines the
    # ends collide, and the Sharpe ratio is what the panel is being read for.
    y = 0.965
    for key, _, color, _, _ in SERIES:
        if key not in sharpes:
            continue
        ax.annotate(f"{sharpes[key]:+.2f}", (0.032, y), xycoords="axes fraction",
                    fontsize=9, color=color, fontweight="bold", va="top")
        y -= 0.072
    ax.annotate("Sharpe", (0.032, y + 0.012), xycoords="axes fraction",
                fontsize=7.5, color=MUTED, va="top")

    for side in ax.spines.values():
        side.set_color(RULE)
        side.set_linewidth(0.9)
    ax.tick_params(colors=AXIS, labelsize=8.5, length=3, width=0.8)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def draw(series, era):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.9))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.845, bottom=0.185,
                        hspace=0.30, wspace=0.14)

    drawn = 0
    for ax, (cls, title) in zip(axes.ravel(), PANELS):
        members = {k[1]: v for k, v in series.items() if k[0] == cls}
        members = {k: v for k, v in members.items()
                   if k in {s[0] for s in SERIES}}
        curves, sharpes = wealth(members)
        if curves is None:
            ax.set_visible(False)
            continue
        draw_panel(ax, curves, sharpes, title)
        drawn += 1

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905),
               ncol=4, frameon=False, fontsize=10, handlelength=2.6,
               columnspacing=2.6)

    fig.suptitle("Is China's shortfall a China result, or an era result?",
                 fontsize=15, color=INK, y=0.975)
    fig.text(0.5, 0.925,
             f"AQR's own published TSMOM factor earned Sharpe {era['paper_era']:+.2f} "
             f"over the paper's sample and {era['post_paper']:+.2f} since — "
             "the same construction, by the same authors, on the same instruments.",
             ha="center", va="top", fontsize=10.5, color=AXIS)

    method = textwrap.fill(
        "Each line is an equal-weighted 12-month time series momentum portfolio over "
        "the instruments of that asset class: long if the past 12-month excess return "
        "was positive, short if negative, sized to 40% annualized volatility, held one "
        "month (Moskowitz, Ooi & Pedersen 2012, Eqs. 5 and 6), gross of costs. Each "
        "panel is cut to the months every one of its series is live, so the lines are "
        "never compared across different eras. China: 71 Ricequant dominant continuous "
        "contracts. United States: the same global instruments through two vendors, "
        "Yahoo (39) and Wind (42) — the gap between the solid and dashed blue lines is "
        "two data feeds disagreeing about identical contracts, and is the yardstick for "
        "reading the gap to the red line. AQR published factor: the authors' own TSMOM "
        "series from 'Time Series Momentum Factors Monthly.xlsx', which unlike the other "
        "three includes a currency sleeve in its all-assets line. Returns are in local "
        "currency; futures returns are excess returns by construction.", width=176)
    fig.text(0.008, 0.135, method, va="top", fontsize=8, color=MUTED,
             linespacing=1.5)

    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    return drawn


def main():
    classes, series = class_table()
    era_rows = classes[(classes["source"] == "aqr")
                       & (classes["asset_class"] == "all")].set_index("window")
    era = {w: era_rows.loc[w, "sharpe"] for w in ("paper_era", "post_paper")}

    drawn = draw(series, era)
    print(f"Saved -> {OUT}   ({drawn} panels)")

    common = classes[classes["window"] == "common"]
    for cls, title in PANELS:
        block = common[common["asset_class"] == cls]
        if block.empty:
            continue
        cells = "   ".join(f"{r['source']:>5} {r['sharpe']:+.2f}"
                           for _, r in block.iterrows())
        print(f"  {title:<26} n={int(block['months'].iloc[0]):>3}   {cells}")
    print(f"  AQR published TSMOM: {era['paper_era']:+.2f} in the paper's sample, "
          f"{era['post_paper']:+.2f} since")


if __name__ == "__main__":
    main()
