"""Paper Fig. 3, redrawn: TSMOM vs the S&P 500, growth of $100 on a log scale.

TSMOM here is TARGET A -- AQR's published factor, the paper's own output
(1985-2009). The paper's Fig. 3 compares TSMOM to a diversified passive long
book; this compares it to the S&P 500 instead.

Both lines are EXCESS returns, i.e. above the T-bill. The AQR factor already is
one; the S&P 500 is a total return, so the T-bill has to come off it, or the
comparison hands the index 25 years of free interest. Labels are in Chinese.
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import _cached, load_aqr_factors

# macOS ships PingFang; without this every Chinese glyph renders as a box.
plt.rcParams["font.sans-serif"] = ["PingFang HK", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False   # keep the minus sign, not a tofu box

START, END = "1985", "2009"


def monthly_rf():
    """Monthly T-bill return from FRED DTB3, back to 1984.

    data.fetch_rf() caches a 2000-onward slice under a fixed key, which is no
    use for a 1985 start, so this pulls its own copy under its own key.
    DTB3 is an annual percent (5.25 = 5.25%/yr); /100 /12 makes it monthly.
    """
    def go():
        import requests
        txt = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3",
            timeout=60).text
        d = pd.read_csv(io.StringIO(txt), parse_dates=[0], index_col=0)
        return pd.to_numeric(d.iloc[:, 0], errors="coerce").dropna().to_frame("DTB3")

    rate = _cached("dtb3_1984", go)["DTB3"] / 100.0 / 12.0
    return rate.resample("ME").mean()


def main():
    tsmom = load_aqr_factors("paper")["TSMOM"].dropna().loc[START:END]

    spx = _cached("gspc", lambda: None).iloc[:, 0].dropna()
    spx_total = (1 + spx.pct_change()).resample("ME").prod() - 1
    spx_excess = spx_total - monthly_rf()

    # Join on year-month, not on the timestamp: AQR stamps each row with the
    # last *trading* day (1985-03-29), while resample("ME") gives the last
    # calendar day (1985-03-31). Matching on dates silently drops ~86 months.
    def by_month(s, label):
        return s.rename(label).set_axis(s.index.to_period("M"))

    curves = pd.concat([by_month(tsmom, "时间序列动量策略 (TSMOM)"),
                        by_month(spx_excess, "标普500指数 (S&P 500)")],
                       axis=1).dropna()

    # Growth of $100, both starting from the month before the sample.
    curves.loc[curves.index[0] - 1] = 0.0
    curves = curves.sort_index()
    curves.index = curves.index.to_timestamp(how="end").normalize()
    wealth = 100 * (1 + curves).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    wealth.plot(ax=ax, logy=True, color=["#C1666B", "#4281A4"], lw=1.6)

    ax.set_title("时间序列动量策略与标普500指数的财富增长对比"
                 "（1985–2009）", fontsize=14, pad=14)
    ax.set_xlabel("年份", fontsize=11)
    ax.set_ylabel("100美元的增长", fontsize=11)
    ax.legend(fontsize=11, loc="upper left", frameon=False)
    ax.grid(True, which="both", axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # Log axis: label the decades in dollars rather than 10^2, 10^3.
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    for name in wealth.columns:
        ax.annotate(f"${wealth[name].iloc[-1]:,.0f}",
                    (wealth.index[-1], wealth[name].iloc[-1]),
                    xytext=(6, -3), textcoords="offset points", fontsize=10)

    plt.tight_layout()
    plt.savefig("fig3_tsmom_vs_spx_cn.png", dpi=150)
    print(f"{curves.index[1].date()} -> {curves.index[-1].date()}  "
          f"{len(curves) - 1} 个月")
    print(wealth.iloc[-1].to_string())
    print("Saved -> fig3_tsmom_vs_spx_cn.png")


if __name__ == "__main__":
    main()
