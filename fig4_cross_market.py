"""
Fig. 4 -- the same asset in two markets. Does time series momentum travel?

The paper's Fig. 2 is one bar per instrument, and the repo now has two of them:
China's 65 contracts and the global 39. Neither can be laid over the other,
because the universes are disjoint and the samples barely overlap. This figure
is the join: 23 underlyings that both markets list, each scored on the months
both legs are live.

PANEL A -- the gap
    A dumbbell per underlying: China's Sharpe ratio and its US counterpart's,
    joined by a connector. Read the LENGTH of the connector, not the position of
    either dot -- the dots are two estimates of the same trade in two places, and
    the question is whether the place matters. Rows carry their sample size
    because TL has 26 shared months and AU has 186, and those are not equal
    evidence no matter how similar the bars look.

PANEL B -- why the gaps are what they are
    Panel A shows that copper agrees and corn does not. Panel B shows why. The x
    axis is the correlation of the two contracts' RAW returns -- are these the
    same asset at all? -- and the y axis is how often their 12-month trend signs
    agree. Copper sits top right: one metal, two exchanges, 96% the same trade.
    Corn sits bottom left: same grain, two segmented markets, agreeing barely
    more often than a coin. Contracts near the bottom left are not evidence about
    momentum travelling; they are evidence that the asset did not travel.

COLOR
    Red is China and blue is the US, in both panels. In Panel B that becomes a
    diverging scale -- red where China's Sharpe is higher, blue where the US
    leg is -- so the hue never changes meaning inside the figure. The palette is
    fig_rq.py's, which clears all six dataviz checks on a white surface
    (worst all-pairs dE 16.2 normal / 12.0 deutan).

Run: python fig4_cross_market.py  ->  fig4_cross_market_pairs_rq.png
"""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from compare_markets import pair_table, source_control_table
from crosswalk import SECTORS

OUT = "fig4_cross_market_pairs_rq.png"

INK = "#1A1A1A"
AXIS = "#4D4D4D"
MUTED = "#6B7178"
RULE = "#C7CCD1"
GRID = "#E4E7EA"

# Same two hues fig_rq.py uses for the commodity and equity sleeves, reused here
# with a single consistent meaning: red is the Chinese leg, blue is the US leg.
CN_COLOR = "#B4504F"
US_COLOR = "#0F72C4"
LINK = "#BFC5CB"        # the connector: recessive, the dots carry identity

# Diverging ramp for Panel B. Warm and cool poles with a neutral -- never a hue
# at the midpoint, so "no difference" reads as nothing.
DIVERGING = LinearSegmentedColormap.from_list(
    "cn_us", [US_COLOR, "#EFEEEB", CN_COLOR])

# Kept short on purpose: these are rotated captions sitting beside blocks as
# small as two rows, and a long one overruns its block into its neighbour.
SECTOR_LABEL = {"metals": "Metals", "energy": "Energy", "ags": "Agriculture",
                "rates": "Bonds", "equity": "Equity"}


def _ordered(pairs):
    """Scored Yahoo rows, grouped by sector, sorted by the gap inside each.

    Returned bottom-up: matplotlib's y axis counts upward, so reversing here
    puts metals at the top of the drawn figure, in SECTORS order.
    """
    rows = pairs[(pairs["us_seam"] == "yahoo") & pairs["d_sharpe"].notna()]
    blocks = [rows[rows["sector"] == s].sort_values("d_sharpe")
              for s in SECTORS]
    return [b for b in reversed(blocks) if not b.empty]


def draw_panel_a(ax, blocks):
    """Horizontal dumbbells, one row per underlying, sector blocks separated."""
    y = 0
    ticks, labels, boundaries = [], [], []
    for block in blocks:
        for _, r in block.iterrows():
            lo, hi = sorted((r["sharpe_cn"], r["sharpe_us"]))
            ax.plot([lo, hi], [y, y], color=LINK, lw=2.4, solid_capstyle="round",
                    zorder=1)
            # A 2px surface ring so the two dots never merge when the gap is small.
            for value, color in ((r["sharpe_us"], US_COLOR),
                                 (r["sharpe_cn"], CN_COLOR)):
                ax.plot(value, y, "o", ms=8.5, color=color, mec="white", mew=1.4,
                        zorder=3)

            proxy = r["tier"] == "proxy"
            ticks.append(y)
            labels.append(f"{r['underlying']}{'  ~' if proxy else ''}")
            # The sample size, set apart on the right. Never a value label on
            # every dot -- the axis already carries the Sharpe ratios.
            ax.annotate(f"{int(r['months'])}", (1.0, y),
                        xycoords=("axes fraction", "data"),
                        xytext=(-4, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=7.5,
                        color=MUTED if not proxy else "#9AA0A6")
            y += 1
        boundaries.append(y - 0.5)
        y += 0.7

    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    for b in boundaries[:-1]:
        ax.axhline(b + 0.35, color=RULE, lw=0.7, zorder=0)

    # Sector names live outside the row labels entirely, in the figure's left
    # margin, so the two label columns can never collide. get_yaxis_transform
    # gives x in axes fractions and y in data coordinates.
    start = 0
    for block, end in zip(blocks, boundaries):
        mid = (start + end - 0.5) / 2
        ax.text(-0.305, mid, SECTOR_LABEL[block["sector"].iloc[0]],
                transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=9, color=INK, fontweight="bold", rotation=90)
        start = end + 1.2

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.9, y - 0.9)
    ax.set_xlabel("Annualized Sharpe ratio, 12-month time series momentum",
                  fontsize=10, color=AXIS, labelpad=8)
    ax.annotate("shared\nmonths", (1.0, 1.0), xycoords="axes fraction",
                xytext=(-4, 8), textcoords="offset points", ha="right",
                va="bottom", fontsize=7.5, color=MUTED, linespacing=1.3)

    # The legend sits above the plot area rather than inside it -- with 23 rows
    # spanning the full width there is no empty corner it could occupy without
    # covering a dumbbell.
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8.5, mec="white", mew=1.4,
                          color=c, label=l)
               for c, l in ((CN_COLOR, "China (Ricequant)"),
                            (US_COLOR, "United States (Yahoo)"))]
    ax.legend(handles=handles, fontsize=9.5, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.002), frameon=False, handletextpad=0.4,
              columnspacing=2.4, borderaxespad=0.0)
    ax.set_title("Each contract against its counterpart in the other market\n"
                 "( ~ marks an economic proxy rather than the same asset )",
                 fontsize=11, color=INK, pad=30)


def _place_labels(ax, rows, sizes, reserved=(), fontsize=7.2):
    """Direct-label every dot it is possible to label without an overlap.

    23 points in one cloud will collide under any single fixed offset, and a
    figure with overlapping labels is unreadable regardless of how good the
    statistics are. Each label tries eight positions around its dot -- above,
    below, right, left, then the diagonals -- and takes the first that clears
    every dot and every label already placed. Anything that clears none is left
    unlabelled rather than drawn on top of a neighbour; its dot is still there,
    and outputs/cross_market_pairs.csv is the table view.
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    def to_ax(x, y):
        return (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)

    # Rough label extent in axes fractions. Deliberately generous: a false
    # collision costs one label, a missed one costs legibility.
    char_w, line_h = 0.0070 * (fontsize / 7.2), 0.030
    pad_x, pad_y = 0.011, 0.020

    boxes = []
    for (_, r), s in zip(rows.iterrows(), sizes):
        cx, cy = to_ax(r["corr_underlying"], r["signal_agree"])
        rad = np.sqrt(s / np.pi) / 460          # marker radius, axes fractions
        boxes.append((cx - rad, cy - rad * 1.7, cx + rad, cy + rad * 1.7))

    placed, skipped = list(boxes) + list(reserved), 0
    for (_, r), s in zip(rows.iterrows(), sizes):
        text = r["underlying"]
        w, h = len(text) * char_w, line_h
        cx, cy = to_ax(r["corr_underlying"], r["signal_agree"])
        rad = np.sqrt(s / np.pi) / 460

        candidates = [
            (0, rad + pad_y, "center", "bottom"),
            (0, -(rad + pad_y), "center", "top"),
            (rad + pad_x, 0, "left", "center"),
            (-(rad + pad_x), 0, "right", "center"),
            (rad * 0.8, rad + pad_y, "left", "bottom"),
            (-rad * 0.8, rad + pad_y, "right", "bottom"),
            (rad * 0.8, -(rad + pad_y), "left", "top"),
            (-rad * 0.8, -(rad + pad_y), "right", "top"),
        ]
        for dx, dy, ha, va in candidates:
            lx, ly = cx + dx, cy + dy
            bx0 = lx if ha == "left" else (lx - w if ha == "right" else lx - w / 2)
            by0 = ly if va == "bottom" else (ly - h if va == "top" else ly - h / 2)
            box = (bx0, by0, bx0 + w, by0 + h)
            if box[0] < -0.02 or box[2] > 1.02 or box[1] < 0 or box[3] > 1.0:
                continue
            if any(not (box[2] < b[0] or box[0] > b[2]
                        or box[3] < b[1] or box[1] > b[3]) for b in placed):
                continue
            ax.text(lx, ly, text, transform=ax.transAxes, ha=ha, va=va,
                    fontsize=fontsize, color=AXIS)
            placed.append(box)
            break
        else:
            skipped += 1
    return skipped


def draw_panel_b(ax, pairs):
    """Is it even the same asset? Return correlation against signal agreement."""
    rows = pairs[(pairs["us_seam"] == "yahoo")
                 & pairs["corr_underlying"].notna()
                 & pairs["signal_agree"].notna()]

    span = max(0.9, rows["d_sharpe"].abs().max())
    norm = Normalize(-span, span)
    sizes = 26 + 240 * (rows["months"] / rows["months"].max())

    # Limits first: _place_labels works in axes fractions and needs them fixed.
    ax.set_ylim(0.30, 1.03)
    ax.set_xlim(-0.26, 1.02)

    ax.axhline(0.5, color=RULE, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.axvline(0, color=RULE, lw=0.9, zorder=0)
    ax.scatter(rows["corr_underlying"], rows["signal_agree"], s=sizes,
               c=[DIVERGING(norm(v)) for v in rows["d_sharpe"]],
               edgecolors="white", linewidths=1.3, zorder=3)

    # The two corner captions are drawn first and passed to the labeller as
    # obstacles, so a contract name can never land on top of one.
    corners = [(0.985, 0.985, "right", "top", "same asset,\nsame trade"),
               (0.015, 0.015, "left", "bottom", "same name,\ndifferent market")]
    for x, y, ha, va, text in corners:
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5,
                color=MUTED, linespacing=1.35, style="italic")
    reserved = [(0.70, 0.90, 1.00, 1.00), (0.00, 0.00, 0.30, 0.10)]
    ax.annotate("coin flip", (0.985, 0.5), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", ha="right",
                va="bottom", fontsize=7.5, color=MUTED)

    skipped = _place_labels(ax, rows, sizes, reserved=reserved)

    ax.set_xlabel("Correlation of the two contracts' monthly excess returns",
                  fontsize=10, color=AXIS, labelpad=8)
    ax.set_ylabel("Months the two 12-month trend signals agree",
                  fontsize=10, color=AXIS, labelpad=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return skipped
    ax.set_title("Whether the two contracts are the same asset at all\n"
                 "(marker area = shared months;  red = China's Sharpe is higher)",
                 fontsize=11, color=INK, pad=10)


def style(ax):
    for side in ax.spines.values():
        side.set_color(RULE)
        side.set_linewidth(0.9)
    ax.tick_params(colors=AXIS, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def draw(pairs, source):
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 8.8),
                             gridspec_kw={"width_ratios": [1.18, 1.0]})
    # Explicit margins rather than tight_layout: the sector captions live
    # outside the axes in the left margin, which tight_layout cannot see.
    fig.subplots_adjust(left=0.152, right=0.985, top=0.855, bottom=0.258,
                        wspace=0.28)

    blocks = _ordered(pairs)
    draw_panel_a(axes[0], blocks)
    skipped = draw_panel_b(axes[1], pairs)
    for ax in axes:
        style(ax)
    axes[0].grid(False, axis="y")

    n = sum(len(b) for b in blocks)
    fig.suptitle("The same asset in two markets: does time series momentum travel?",
                 fontsize=15, color=INK, y=0.985)

    scored = source[source["d_sharpe_source"].notna()]
    market_gap = pairs[pairs["us_seam"] == "yahoo"]["d_sharpe"].abs().median()
    control = (
        f"Measurement control: the same {len(scored)} US contracts priced by two "
        f"different vendors (Yahoo and Wind) disagree about their own Sharpe ratio "
        f"by a median of {scored['d_sharpe_source'].abs().median():.2f} and agree on "
        f"the trend sign {scored['signal_agree_source'].median():.0%} of months. "
        f"Across markets the median gap is {market_gap:.2f} at "
        f"{pairs[pairs['us_seam'] == 'yahoo']['signal_agree'].median():.0%} agreement. "
        "The cross-market difference is real but only modestly larger than the "
        "difference between two data feeds quoting the same contract.")

    method = textwrap.fill(
        f"{n} underlyings listed in both markets, mapped in crosswalk.py. Each leg is "
        "that contract's own 12-month time series momentum strategy: long if the past "
        "12-month excess return was positive, short if negative, sized to 40% "
        "annualized volatility, held one month (Moskowitz, Ooi & Pedersen 2012, Eq. 5), "
        "gross of costs. Every statistic is computed only on the months BOTH legs are "
        "live, so no pair is scored against an era the other side never traded. Returns "
        "stay in local currency -- these are futures excess returns, so converting would "
        "add an FX carry trade to the Chinese leg that the paper never had; the "
        "consequence is that the return correlations in the right panel are attenuated "
        "by the CNY/USD move. China: Ricequant dominant continuous contracts, 2010-2026. "
        "United States: Yahoo continuous contracts, 2000-2026.", width=196)

    fig.text(0.008, 0.196, textwrap.fill(control, width=196) + "\n\n" + method,
             va="top", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    return skipped


def main():
    pairs = pair_table()
    source = source_control_table()
    skipped = draw(pairs, source)
    if skipped:
        print(f"  note: {skipped} scatter labels omitted (no collision-free slot)")

    scored = pairs[(pairs["us_seam"] == "yahoo") & pairs["d_sharpe"].notna()]
    print(f"Saved -> {OUT}   ({len(scored)} scored pairs)")
    ahead = int((scored["d_sharpe"] > 0).sum())
    print(f"  China's leg is ahead on {ahead} of {len(scored)}; "
          f"{int((scored['t_d_sharpe'].abs() > 1.96).sum())} of the gaps clear |z| > 1.96")
    top = scored.reindex(scored["corr_underlying"].sort_values(ascending=False).index)
    print("  most integrated:  " + ", ".join(
        f"{r['underlying']} ({r['corr_underlying']:.2f}/{r['signal_agree']:.0%})"
        for _, r in top.head(3).iterrows()))
    print("  least integrated: " + ", ".join(
        f"{r['underlying']} ({r['corr_underlying']:.2f}/{r['signal_agree']:.0%})"
        for _, r in top.tail(3).iterrows()))


if __name__ == "__main__":
    main()
