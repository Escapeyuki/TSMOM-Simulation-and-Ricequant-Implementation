"""
The paper's Fig. 1 and Table 2, run on China futures.

This is the evidence the rest of the paper rests on. Before building any
portfolio, Moskowitz, Ooi & Pedersen ask a narrower question: does a contract's
own return h months ago predict its return today? They answer it with two pooled
panel regressions (Section 3.1), and the answer is the shape of Fig. 1 --
positive coefficients out to about 12 months, then negative ones as the trend
reverses.

    Eq. 2:  r_t / sigma_{t-1}  =  a + b_h * ( r_{t-h} / sigma_{t-h-1} )  +  e
    Eq. 3:  r_t / sigma_{t-1}  =  a + b_h * sign( r_{t-h} )             +  e

Both sides are divided by the instrument's own ex-ante volatility. The paper's
reason (Section 3.1) is that volatilities differ enormously across contracts --
a bond future and a natural gas future are not on the same scale -- and scaling
puts every observation in the same units. It is, in their words, "similar to
using Generalized Least Squares instead of Ordinary Least Squares".

THE STANDARD ERRORS ARE THE WHOLE GAME HERE
    Stacking ~70 contracts x ~190 months gives around 13,000 rows, and it is
    tempting to treat that as 13,000 independent observations. It is not: in any
    given month, the contracts move together. Plain OLS standard errors would
    assume away that dependence and report t-statistics several times too large.
    The paper clusters by month, so this file does too, via the `cluster`
    argument added to replicate.ols(). Run it without clustering and the
    significance roughly triples -- which is exactly why the paper says so.

TABLE 2 IS A DIFFERENT QUESTION
    Fig. 1 asks whether the past predicts the future. Table 2 asks whether a
    tradeable strategy built on that prediction earns an alpha, for every
    combination of look-back and holding period. The paper's headline choice,
    12 months look-back and 1 month holding, is one cell of that grid, and the
    grid exists to show that the result is not an artifact of picking that cell.
"""

import numpy as np
import pandas as pd

from data_rq import ASSET_CLASS
from factors_rq import monthly_factors
from replicate import ols
from replicate_rq import build as rq_build
from tsmom import VOL_TARGET, build_tsmom

# Fig. 1 runs the regression at every monthly lag out to five years.
MAX_LAG = 60

# Table 2's grid, both axes, exactly the paper's.
PERIODS = [1, 3, 6, 9, 12, 24, 36, 48]

CLASSES = ["commodity", "equity", "bond"]

OUT = "fig1_tsmom_predictability_rq.png"


def scaled_panels(panel, parts=None):
    """The two ingredients of Eq. 2: monthly returns and lagged volatility.

    Returns (monthly_ret, vol_lagged) where vol_lagged[t] is the ex-ante
    annualized volatility measured at the end of month t-1 -- the quantity the
    paper is careful to use so that nothing is scaled by information from its
    own month.

    `parts` is build_tsmom's third return value; pass it in if the caller
    already has one rather than paying for the whole engine a second time.
    """
    if parts is None:
        _, _, parts = build_tsmom(panel)
    monthly_ret = parts["monthly_ret"]
    vol_lagged = parts["vol_m"].shift(1)

    # A month in which an instrument did not trade compounds to 1.0 - 1.0 = 0.0
    # rather than NaN, because the product of no numbers is one. Left alone that
    # would feed the regression a stream of fake zero returns from before each
    # contract was listed. Blank them wherever there was no price at all.
    traded = panel.notna().resample("ME").sum() > 0
    monthly_ret = monthly_ret.where(traded)

    return monthly_ret, vol_lagged


def stack_lag(monthly_ret, vol_lagged, lag, use_sign):
    """Build the pooled (y, x, month) arrays for one lag.

    y is always the volatility-scaled current return. x is either the
    volatility-scaled lagged return (Eq. 2) or just its sign (Eq. 3). In the sign
    version the right-hand side is left unscaled because a sign is already +1 or
    -1 and dividing it by a volatility would undo the point of it.
    """
    y = (monthly_ret / vol_lagged).stack(future_stack=True).rename("y")

    if use_sign:
        x = np.sign(monthly_ret.shift(lag)).stack(future_stack=True).rename("x")
    else:
        x = (monthly_ret.shift(lag) / vol_lagged.shift(lag)).stack(future_stack=True).rename("x")

    both = pd.concat([y, x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if use_sign:
        both = both[both["x"] != 0.0]      # a flat 12 months is not a direction

    months = both.index.get_level_values(0)
    return both["y"].to_numpy(), both["x"].to_numpy(), months.to_numpy()


def lag_curve(monthly_ret, vol_lagged, use_sign, max_lag=MAX_LAG):
    """Run the pooled regression once per lag and collect the t-statistics."""
    rows = {}
    for lag in range(1, max_lag + 1):
        y, x, months = stack_lag(monthly_ret, vol_lagged, lag, use_sign)
        if len(y) < 100:
            continue
        table, _ = ols(y, [x], ["beta"], cluster=months)
        rows[lag] = {"coef": table.loc["beta", "coef"],
                     "t": table.loc["beta", "t"],
                     "n": len(y)}
    return pd.DataFrame(rows).T


def tsmom_jh(monthly_ret, vol_m, lookback, holding):
    """The paper's TSMOM(j, h) strategy return, Section 3.2.

    For a holding period longer than one month the paper does not simply hold a
    position for h months. It runs h overlapping portfolios side by side -- one
    opened in each of the last h months -- and reports the average return of all
    of them. So in any month you own a blend of the position you opened
    yesterday and the ones you opened up to h months ago, each still running out
    its holding period.

    That construction is what makes the resulting series a genuine monthly
    portfolio return rather than an overlapping-window artifact, which in turn is
    why a plain t-statistic on its alpha is legitimate.
    """
    signal = np.sign(np.log1p(monthly_ret).rolling(lookback).sum())
    size = VOL_TARGET / vol_m
    target = signal * size

    legs = []
    for age in range(holding):
        # The portfolio opened `age` months before last month is still running,
        # and earns this month's return on the position it fixed back then.
        position = target.shift(1 + age)
        legs.append((position * monthly_ret).mean(axis=1))

    return pd.concat(legs, axis=1).mean(axis=1)


def alpha_grid(monthly_ret, vol_m, factors, months=None):
    """Table 2 -- the t-statistic of alpha for every look-back x holding pair.

    `months` restricts the sample to the breadth-filtered window replicate_rq.py
    reports on, so the two files are describing the same thing. Without it the
    grid includes 2010-2011, where the portfolio is an average over three or four
    commodity contracts and its variance swamps everything.
    """
    grid = pd.DataFrame(index=PERIODS, columns=PERIODS, dtype=float)
    raw = pd.DataFrame(index=PERIODS, columns=PERIODS, dtype=float)
    for lookback in PERIODS:
        for holding in PERIODS:
            strategy = tsmom_jh(monthly_ret, vol_m, lookback, holding)
            if months is not None:
                strategy = strategy.reindex(months)
            joined = pd.concat([strategy.rename("r"), factors], axis=1).dropna()
            if len(joined) < 36:
                continue
            table, _ = ols(joined["r"], [joined[c] for c in factors.columns],
                           list(factors.columns))
            grid.loc[lookback, holding] = table.loc["intercept", "t"]

            # The same cell with no controls at all: a one-sample t-test on the
            # strategy's own mean return. See the note in main() for why both
            # numbers are reported instead of just the alpha.
            r = joined["r"]
            raw.loc[lookback, holding] = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    for frame in (grid, raw):
        frame.index.name = "lookback"
        frame.columns.name = "holding"
    return grid, raw


def show_curve(label, curve):
    """Print the shape of a lag curve compactly -- the sign pattern is the point."""
    first = curve.loc[1:12, "t"]
    later = curve.loc[13:, "t"]
    print(f"  {label:<22} "
          f"lags 1-12: {int((first > 0).sum())}/12 positive, "
          f"{int((first > 1.96).sum())} significant   |   "
          f"lags 13-60: {int((later < 0).sum())}/{len(later)} negative, "
          f"{int((later < -1.96).sum())} significant")


def validate_on_wind():
    """Run the identical regression on the paper's OWN asset universe.

    This is the control, and it is the reason the China result can be reported as
    a finding rather than a suspicion. A null result is ambiguous by nature: if
    lags 1-12 come out flat, either China has no time series momentum or this
    file computes the paper's regression wrongly. Nothing in the China output can
    tell those apart.

    So the same code is pointed at data_wind.py's 42 global futures -- the
    paper's kind of instrument -- and then at the pre-2010 slice of them, which
    is as close to the paper's 1985-2009 window as this repo can get. If the
    pattern appears there, the machinery works and the China result is about
    China.

    It is the same argument replicate.py makes when it runs the straddle test on
    AQR's own factor over AQR's own window before running it on anything else.
    """
    print("=" * 100)
    print("CONTROL -- the same regression on global futures (data_wind.py)")
    print("=" * 100)

    try:
        from data_wind import futures_panel as wind_panel
        panel = wind_panel()
    except Exception as exc:
        print(f"  skipped: {exc}")
        return

    print(f"  {panel.shape[1]} contracts, {panel.index[0].date()} -> "
          f"{panel.index[-1].date()}\n")

    monthly_ret, vol_lagged = scaled_panels(panel)
    show_curve("global, full sample", lag_curve(monthly_ret, vol_lagged, use_sign=False))

    # The pre-2010 slice: the paper's era, or as near as this data reaches.
    early = panel.loc[:"2009-12-31"]
    m_early, v_early = scaled_panels(early)
    curve = lag_curve(m_early, v_early, use_sign=False)
    show_curve("global, 1990-2009", curve)
    print("      lags 1-12 t: " + "  ".join(f"{v:+5.2f}" for v in curve.loc[1:12, "t"]))
    print("\n  The paper reports 12/12 positive and nine significant over 1985-2009.")
    print("  If this control shows the pattern and the China panel does not, the")
    print("  difference is the market, not the code.\n")


def main():
    # One trip through the engine, shared by everything below. replicate_rq.build
    # runs it too, which is why its results come back from the same call rather
    # than from a second one.
    panel, div, per_inst, parts, _ = rq_build()
    monthly_ret, vol_lagged = scaled_panels(panel, parts)
    vol_m = parts["vol_m"]

    print("=" * 100)
    print("FIG. 1 -- time series predictability, pooled panel, SEs clustered by month")
    print("=" * 100)
    print(f"  {panel.shape[1]} contracts, {monthly_ret.index[0].date()} -> "
          f"{monthly_ret.index[-1].date()}\n")

    print("  The paper's finding: positive out to 12 months, then reversal.")
    print("  (Panel A of its Fig. 1 shows 12/12 positive lags, nine significant.)\n")

    panel_a = lag_curve(monthly_ret, vol_lagged, use_sign=False)
    panel_b = lag_curve(monthly_ret, vol_lagged, use_sign=True)
    show_curve("Panel A (Eq. 2)", panel_a)
    show_curve("Panel B (Eq. 3, sign)", panel_b)

    print("\n  Panel A t-statistics by lag:")
    for lo in range(1, 61, 12):
        chunk = panel_a.loc[lo:lo + 11, "t"]
        print(f"    lags {lo:>2}-{lo + 11:<2}  " +
              "  ".join(f"{v:+5.2f}" for v in chunk))

    # Panel C -- the same regression one asset class at a time.
    print("\n  Panel C -- by asset class (Eq. 2):")
    for cls in CLASSES:
        cols = [c for c in panel.columns if ASSET_CLASS[c] == cls]
        curve = lag_curve(monthly_ret[cols], vol_lagged[cols], use_sign=False)
        show_curve(f"  {cls} ({len(cols)})", curve)

    # What the clustering is worth, stated rather than asserted.
    y, x, months = stack_lag(monthly_ret, vol_lagged, 1, use_sign=False)
    clustered, _ = ols(y, [x], ["beta"], cluster=months)
    plain, _ = ols(y, [x], ["beta"])
    print(f"\n  clustering check at lag 1: t = {clustered.loc['beta', 't']:+.2f} "
          f"clustered vs {plain.loc['beta', 't']:+.2f} unclustered "
          f"({len(y)} rows, {len(np.unique(months))} months)")
    print("  The unclustered number is the one that would be wrong.")

    # Before reading anything above as a statement about China, check that the
    # regression reproduces the paper where the paper's own data is available.
    print()
    validate_on_wind()

    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE 2 -- t-statistic of alpha, by look-back (rows) and holding period (cols)")
    print("=" * 100)
    factors = monthly_factors()
    grid, raw = alpha_grid(monthly_ret, vol_m, factors, months=div.index)
    print(f"  alpha from r = a + b1*MKT + b2*BOND + b3*GSCI + s*SMB + h*HML + m*UMD")
    print(f"  (paper Eq. 24, with the China substitutions listed in factors_rq.py)\n")
    print(grid.round(2).to_string())

    # Both grids are printed because the control factors here are substitutes,
    # and two of them -- SMB at a Sharpe of 1.6 and the bond premium at 1.7 --
    # are far stronger in China over this window than their US counterparts were
    # over the paper's. Regressing on a high-Sharpe factor shrinks alpha even
    # when the loading is insignificant, so a low alpha t-statistic here is
    # partly a statement about the controls rather than about TSMOM. The raw
    # grid asks the narrower question the controls cannot distort: did the
    # strategy make money at all?
    print("\n  the same grid with NO controls -- t-statistic of the mean return:\n")
    print(raw.round(2).to_string())

    print(f"\n  paper Panel A, 12-month look-back / 1-month hold: t = 6.61")
    print(f"  here:  alpha t = {grid.loc[12, 1]:.2f}    "
          f"raw mean t = {raw.loc[12, 1]:.2f}")

    _plot(panel_a, panel_b)


def _plot(panel_a, panel_b):
    """Fig. 1 -- t-statistic by lag, the paper's signature picture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, curve, title in [
        (axes[0], panel_a, "Panel A: $r_t/\\sigma_{t-1}$ on $r_{t-h}/\\sigma_{t-h-1}$"),
        (axes[1], panel_b, "Panel B: $r_t/\\sigma_{t-1}$ on $sign(r_{t-h})$"),
    ]:
        colors = ["#4281A4" if v > 0 else "#C1666B" for v in curve["t"]]
        ax.bar(curve.index, curve["t"], color=colors, width=0.8)
        # +-1.96 is the 5% two-sided cutoff; the paper's own figures show it.
        for level in (1.96, -1.96):
            ax.axhline(level, color="#4D4D4D", lw=0.8, ls=":")
        ax.axhline(0, color="#1A1A1A", lw=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Month lag")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("t-statistic (clustered by month)")

    fig.suptitle("Time series predictability of China futures, 2010-2026 "
                 "(paper Fig. 1, out of sample)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
