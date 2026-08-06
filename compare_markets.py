"""
The same asset, two markets, three data feeds: does time series momentum travel?

WHAT THIS ASKS
    fig2_sharpe_by_instrument_rq.png says China's TSMOM works on 43 of 65
    contracts. The right panel of tsmom_replication.png says the global version
    works on 28 of 39. Those are two different claims about two different
    universes over two different periods, and neither answers the question a
    reader actually has: take gold -- the same metal, sized the same way, traded
    in Shanghai and in New York over the same months -- does the strategy do the
    same thing in both places?

    crosswalk.py says which contracts are the same asset. This file computes the
    comparison.

THREE THINGS THIS SEPARATES THAT A NAIVE COMPARISON CONFOUNDS
    1. Era.    China's data starts 2010; the paper's ends 2009. Any CN-vs-US
               difference measured on each side's own full sample is partly a
               statement about 2010-2026 versus 1985-2009. Every headline number
               here is computed on the months where BOTH legs are live, and the
               full-sample figures are carried alongside as context only.
    2. Sample. TL (30-year CGB) has 27 shared months; AU has 186. Those are not
               equal evidence, so `months` travels with every row and the
               figures annotate it.
    3. Feed.   Yahoo and Wind are two vendors quoting the same COMEX gold. If
               they disagree about its Sharpe ratio by as much as Shanghai and
               COMEX disagree, then the cross-market gap is measurement, not
               economics. That control is computed here as its own table.

THE FOUR STATISTICS, AND WHY EACH
    sharpe_cn / sharpe_us    the headline, Eq. 5 run per instrument
    d_sharpe + t_d_sharpe    the difference, tested properly. Two Sharpe ratios
                             on correlated return streams cannot be compared
                             with independent t-tests; this uses the
                             Jobson-Korkie statistic with Memmel's correction.
    corr_underlying          correlation of the two RAW monthly excess returns.
                             This is the market-integration measure and it has
                             nothing to do with momentum: it asks whether the
                             two contracts are the same asset at all.
    signal_agree             fraction of months where the two 12-month trend
                             signs agree. This is the momentum-specific version
                             of the same question, and it is the one that
                             explains the Sharpe gaps: copper agrees 96% of the
                             time, corn 64%.

CURRENCY
    Everything stays in local currency. These are futures excess returns, so the
    CNY/USD move belongs to the funding leg, not the strategy leg; converting
    would add an FX carry trade to the Chinese side that the paper never had.
    The consequence is that corr_underlying is attenuated by the CNY move, and
    that is stated on every figure that uses it.

RUN
    python compare_markets.py           # tables to stdout, CSVs to outputs/
    python compare_markets.py --check   # validation only, no output files
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import crosswalk
from crosswalk import AQR_SLEEVE, PAIRS, SEAMS
from data import load_aqr_factors
from replicate import ols
from tsmom import MONTHS_PER_YEAR, build_tsmom, performance

OUTDIR = Path(__file__).resolve().parent / "outputs"

# Below this many overlapping months a pair is reported but not scored. Same
# threshold replicate_rq.sharpe_t_stats uses to decide an instrument is old
# enough to have an opinion about.
MIN_MONTHS = 24

# The AQR published factors run to 2026-05; the seams run past it. Rollups
# against AQR are cut to the common span rather than letting the last two
# months of one series compare against nothing.
AQR_WHICH = "updated"

_BUILT = {}


# ----------------------------------------------------------------------------
# Seam plumbing
# ----------------------------------------------------------------------------

def seam(name):
    """Build one seam once: panel, diversified factor, per-instrument, parts.

    Deliberately builds on the FULL panel and slices afterwards. Windowing the
    daily panel first would restart the 261-day volatility burn-in and the
    12-month lookback inside the comparison window, which silently changes the
    strategy being compared.
    """
    if name not in _BUILT:
        module = SEAMS[name][0]
        panel = module.futures_panel()
        div, per_inst, parts = build_tsmom(panel)
        _BUILT[name] = {
            "panel": panel, "div": div, "per_inst": per_inst, "parts": parts,
            "module": module, "label": SEAMS[name][1],
            # Months in which the contract genuinely traded. See _traded().
            "traded": panel.notna().resample("ME").sum() > 0,
        }
    return _BUILT[name]


def _traded(seam_name, code, series):
    """Blank the months a contract booked a return without trading.

    resample("ME").prod() over a month with no data returns 1.0 -- the product
    of an empty set -- so monthly_ret comes out 0.0 rather than NaN, and .ewm()
    carries the last volatility estimate across the gap, so the position is
    non-NaN too. The result is a dead contract quietly booking 0.0% every month
    forever. replicate_rq.breadth_of documents this and leaves it alone because
    on the China panel it is 13 cells out of 8,143, all fuel oil.

    It cannot be left alone here. Wind's Nasdaq series stops in June 2015 and
    then books 133 ghost months, which is a zero-variance return stream: its
    Sharpe ratio is 0/0 and its signal agreement with anything is 0%. Masking
    costs 13 cells on the China seam and 15 on Yahoo, and fixes the one pair
    that was otherwise nonsense.

    panel_regressions.scaled_panels applies the same mask for the same reason.
    """
    mask = seam(seam_name)["traded"]
    if code not in mask.columns:
        return series
    return series.where(mask[code].reindex(series.index).fillna(False))


def as_period(frame_or_series):
    """Re-index month-end timestamps to periods so seams and AQR can be joined.

    resample("ME") stamps the calendar month end; AQR stamps the last trading
    day. Joining those on timestamps silently drops most of the sample -- the
    bug fig3_wind.py and fig_rq.py both had to work around.
    """
    out = frame_or_series.copy()
    out.index = out.index.to_period("M")
    return out


def leg(seam_name, code, kind="per_inst"):
    """One instrument's monthly series from one seam, period-indexed.

    kind: "per_inst"    the TSMOM strategy return (Eq. 5)
          "monthly_ret" the raw monthly excess return of the contract
          "signal"      sign of the trailing 12-month return
    """
    built = seam(seam_name)
    frame = built["per_inst"] if kind == "per_inst" else built["parts"][kind]
    if code is None or code not in frame.columns:
        return None
    return as_period(_traded(seam_name, code, frame[code])).dropna()


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def sharpe(r):
    sd = r.std(ddof=1)
    return r.mean() * MONTHS_PER_YEAR / (sd * np.sqrt(MONTHS_PER_YEAR)) if sd else np.nan


def tstat(r):
    sd = r.std(ddof=1)
    return r.mean() / (sd / np.sqrt(len(r))) if sd else np.nan


def jk_memmel(a, b):
    """Test that two Sharpe ratios computed on the SAME months differ.

    Jobson & Korkie (1981) with Memmel's (2003) correction. Two independent
    t-tests would be wrong here: Shanghai gold momentum and COMEX gold momentum
    are the same trade in two places and their returns are correlated, so the
    difference is estimated far more precisely than independent tests assume.

        theta = (1/T) [ 2(1-rho) + 0.5 (SR_a^2 + SR_b^2 - 2 SR_a SR_b rho^2) ]
        z     = (SR_a - SR_b) / sqrt(theta)

    Computed on per-month Sharpe ratios (annualising both sides cancels).
    Returns z, which is standard normal under the null of equal Sharpes.
    """
    n = len(a)
    if not (a.std(ddof=1) > 0 and b.std(ddof=1) > 0):
        return np.nan
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = a.corr(b)
    theta = (2 * (1 - rho) + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / n
    return (sa - sb) / np.sqrt(theta) if theta > 0 else np.nan


def spanning(a, b):
    """Regress leg a on leg b. Returns (alpha per month, t(alpha), beta, R^2).

    "Does the Chinese contract's momentum earn anything its US twin does not?"
    A positive significant alpha means the Shanghai leg is not redundant.
    """
    table, r2 = ols(a.to_numpy(), [b.to_numpy()], ["us"])
    return (table.loc["intercept", "coef"], table.loc["intercept", "t"],
            table.loc["us", "coef"], r2)


def _pair_stats(cn_leg, us_leg, cn_code, us_code, cn_seam, us_seam):
    """Every statistic for one (cn, us) pair, on their intersected months."""
    joined = pd.concat([cn_leg.rename("cn"), us_leg.rename("us")], axis=1).dropna()
    out = {
        "months": len(joined),
        "start": str(joined.index[0]) if len(joined) else None,
        "end": str(joined.index[-1]) if len(joined) else None,
    }
    if len(joined) < MIN_MONTHS:
        return out

    a, b = joined["cn"], joined["us"]
    out.update({
        "sharpe_cn": sharpe(a), "t_cn": tstat(a),
        "sharpe_us": sharpe(b), "t_us": tstat(b),
        "d_sharpe": sharpe(a) - sharpe(b),
        "t_d_sharpe": jk_memmel(a, b),
        "corr_tsmom": a.corr(b),
        "ann_mean_cn": a.mean() * MONTHS_PER_YEAR,
        "ann_mean_us": b.mean() * MONTHS_PER_YEAR,
    })

    alpha, t_alpha, beta, r2 = spanning(a, b)
    out.update({"alpha_span": alpha, "t_alpha_span": t_alpha,
                "beta_span": beta, "r2_span": r2})

    # The two integration measures, on the identical window.
    window = joined.index
    ret = pd.concat([leg(cn_seam, cn_code, "monthly_ret").rename("cn"),
                     leg(us_seam, us_code, "monthly_ret").rename("us")],
                    axis=1).reindex(window).dropna()
    out["corr_underlying"] = ret["cn"].corr(ret["us"]) if len(ret) > 12 else np.nan

    sig = pd.concat([leg(cn_seam, cn_code, "signal").rename("cn"),
                     leg(us_seam, us_code, "signal").rename("us")],
                    axis=1).reindex(window).dropna()
    out["signal_agree"] = (sig["cn"] == sig["us"]).mean() if len(sig) > 12 else np.nan
    out["signal_months"] = len(sig)
    return out


# ----------------------------------------------------------------------------
# The three tables
# ----------------------------------------------------------------------------

def pair_table():
    """One row per (underlying, US seam). The headline table."""
    rows = []
    for underlying, sector, cn_code, yahoo, wind, tier, note in PAIRS:
        cn_leg = leg("cn", cn_code)
        for us_seam, us_code in (("yahoo", yahoo), ("wind", wind)):
            if us_code is None:
                continue
            us_leg = leg(us_seam, us_code)
            if cn_leg is None or us_leg is None:
                continue

            row = {
                "underlying": underlying, "sector": sector, "tier": tier,
                "asset_class": crosswalk.asset_class("cn", cn_code),
                "us_seam": us_seam,
                "cn": cn_code, "cn_name": crosswalk.name_of("cn", cn_code),
                "us": us_code, "us_name": crosswalk.name_of(us_seam, us_code),
            }
            row.update(_pair_stats(cn_leg, us_leg, cn_code, us_code, "cn", us_seam))
            # Each leg on its own full history, for context only.
            row.update({
                "sharpe_cn_full": sharpe(cn_leg), "months_cn_full": len(cn_leg),
                "sharpe_us_full": sharpe(us_leg), "months_us_full": len(us_leg),
                "note": note,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def source_control_table():
    """Yahoo vs Wind on the SAME US contract -- the measurement-error control.

    Whatever this table shows is the floor: a CN-vs-US gap smaller than the
    Yahoo-vs-Wind gap for the same underlying is not evidence about markets.
    """
    rows = []
    for underlying, sector, cn_code, yahoo, wind, tier, _ in PAIRS:
        if yahoo is None or wind is None:
            continue
        y, w = leg("yahoo", yahoo), leg("wind", wind)
        if y is None or w is None:
            continue
        joined = pd.concat([y.rename("y"), w.rename("w")], axis=1).dropna()
        row = {"underlying": underlying, "sector": sector, "tier": tier,
               "yahoo": yahoo, "wind": wind, "months": len(joined)}
        if len(joined) >= MIN_MONTHS:
            a, b = joined["y"], joined["w"]
            ret = pd.concat([leg("yahoo", yahoo, "monthly_ret").rename("y"),
                             leg("wind", wind, "monthly_ret").rename("w")],
                            axis=1).reindex(joined.index).dropna()
            sig = pd.concat([leg("yahoo", yahoo, "signal").rename("y"),
                             leg("wind", wind, "signal").rename("w")],
                            axis=1).reindex(joined.index).dropna()
            row.update({
                "sharpe_yahoo": sharpe(a), "sharpe_wind": sharpe(b),
                "d_sharpe_source": sharpe(a) - sharpe(b),
                "t_d_sharpe_source": jk_memmel(a, b),
                "corr_tsmom_source": a.corr(b),
                "corr_underlying_source": ret["y"].corr(ret["w"]),
                "signal_agree_source": (sig["y"] == sig["w"]).mean(),
                "start": str(joined.index[0]), "end": str(joined.index[-1]),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def per_inst_masked(seam_name):
    """per_inst with the ghost months blanked. Cached on the seam."""
    built = seam(seam_name)
    if "per_inst_masked" not in built:
        per = built["per_inst"]
        built["per_inst_masked"] = per.where(
            built["traded"].reindex(per.index).reindex(columns=per.columns).fillna(False))
    return built["per_inst_masked"]


def _sleeve(seam_name, cls):
    """Equal-weighted TSMOM across every instrument of one asset class."""
    built = seam(seam_name)
    per = per_inst_masked(seam_name)
    cols = [c for c in per.columns if built["module"].ASSET_CLASS.get(c) == cls]
    if not cols:
        return None
    return as_period(per[cols].mean(axis=1)).dropna()


def _all_sleeve(seam_name):
    """The diversified factor -- every instrument, breadth unfiltered."""
    return as_period(per_inst_masked(seam_name).mean(axis=1)).dropna()


def class_table():
    """Sleeve-level comparison, including AQR's own published factors.

    Instrument-level matching runs out at 23 pairs; China lists rebar and the US
    does not, the US trades wheat and China's is illiquid. At sleeve level the
    objects stay comparable, and AQR's published TSMOM^CM / ^EQ / ^FI let the
    paper's authors' own construction sit in the same table as ours.

    Emitted long: one row per (asset_class, source, window) so the era split is
    a filter rather than a wider table.
    """
    aqr = as_period(load_aqr_factors(AQR_WHICH))
    series = {}
    for cls in ("commodity", "equity", "bond", "currency"):
        for seam_name in SEAMS:
            s = _sleeve(seam_name, cls)
            if s is not None and len(s) >= MIN_MONTHS:
                series[(cls, seam_name)] = s
        col = AQR_SLEEVE[cls]
        if col in aqr.columns:
            series[(cls, "aqr")] = aqr[col].dropna()
    for seam_name in SEAMS:
        series[("all", seam_name)] = _all_sleeve(seam_name)
    series[("all", "aqr")] = aqr["TSMOM"].dropna()

    rows = []
    for cls in ("all", "commodity", "equity", "bond", "currency"):
        members = {k[1]: v for k, v in series.items() if k[0] == cls}
        if not members:
            continue
        # The window every member of this class shares.
        common = None
        for s in members.values():
            common = s.index if common is None else common.intersection(s.index)

        for source, s in members.items():
            for window, sub in (("full", s),
                                ("common", s.reindex(common).dropna())):
                if len(sub) < MIN_MONTHS:
                    continue
                st = performance(sub)
                rows.append({
                    "asset_class": cls, "source": source, "window": window,
                    "sharpe": st["sharpe"], "t": tstat(sub),
                    "ann_mean": st["ann_mean"], "ann_vol": st["ann_vol"],
                    "max_drawdown": st["max_drawdown"],
                    "months": len(sub),
                    "start": str(sub.index[0]), "end": str(sub.index[-1]),
                })

        # The era control: AQR's own factor, split at the end of the paper's
        # sample. This is what separates "China is different" from "2010-2026
        # is different", and it is the single most important row here.
        if (cls, "aqr") in series:
            a = series[(cls, "aqr")]
            for window, sub in (("paper_era", a.loc[:"2009-12"]),
                                ("post_paper", a.loc["2010-01":])):
                if len(sub) < MIN_MONTHS:
                    continue
                st = performance(sub)
                rows.append({
                    "asset_class": cls, "source": "aqr", "window": window,
                    "sharpe": st["sharpe"], "t": tstat(sub),
                    "ann_mean": st["ann_mean"], "ann_vol": st["ann_vol"],
                    "max_drawdown": st["max_drawdown"], "months": len(sub),
                    "start": str(sub.index[0]), "end": str(sub.index[-1]),
                })

    return pd.DataFrame(rows), series


def class_correlations(series):
    """Correlation of each class's sleeves across seams, on shared months."""
    out = {}
    for cls in ("all", "commodity", "equity", "bond"):
        members = {k[1]: v for k, v in series.items() if k[0] == cls}
        if len(members) < 2:
            continue
        frame = pd.concat(members, axis=1).dropna()
        if len(frame) >= MIN_MONTHS:
            out[cls] = frame.corr()
    return out


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def check():
    """Fail loudly on anything that would quietly produce a wrong row."""
    crosswalk.validate()
    print("  crosswalk.validate() ok -- every code exists in its seam")

    problems = []
    for underlying, _, cn_code, yahoo, wind, tier, _ in PAIRS:
        cn_leg = leg("cn", cn_code)
        if cn_leg is None:
            problems.append(f"{underlying}: {cn_code} has no strategy series in cn")
            continue
        for seam_name, code in (("yahoo", yahoo), ("wind", wind)):
            if code is None:
                continue
            other = leg(seam_name, code)
            if other is None:
                problems.append(f"{underlying}: {code} has no series in {seam_name}")
                continue
            shared = cn_leg.index.intersection(other.index)
            flag = "" if len(shared) >= MIN_MONTHS else "  (under 24 -- reported unscored)"
            if tier == "exact" and seam_name == "yahoo":
                print(f"  {underlying:<22} {cn_code:>3} vs {code:<8} "
                      f"{len(shared):>4} shared months{flag}")

    # The collision guard: these codes mean different things in different seams,
    # so any code-only lookup is a latent bug.
    for code in ("B", "LC", "SI", "C"):
        seen = [s for s in SEAMS if code in crosswalk.universe(s)]
        if len(seen) > 1:
            problems.append(f"bare code {code!r} exists in more than one seam: {seen}")
    print(f"  collision guard: bare codes B/LC/SI/C are seam-local as expected")

    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    {p}")
        return False
    print("\n  all checks passed")
    return True


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def _show_pairs(pairs):
    scored = pairs[(pairs["us_seam"] == "yahoo") & pairs["sharpe_cn"].notna()]
    scored = scored.sort_values(["sector", "d_sharpe"], ascending=[True, False])
    print(f"\n  {'underlying':<22}{'tier':<7}{'n':>4}  "
          f"{'S_cn':>6}{'S_us':>7}{'diff':>7}{'z':>6}   "
          f"{'corr_ret':>9}{'agree':>7}   {'alpha%/mo':>10}{'t':>6}")
    print("  " + "-" * 100)
    for _, r in scored.iterrows():
        print(f"  {r['underlying']:<22}{r['tier']:<7}{int(r['months']):>4}  "
              f"{r['sharpe_cn']:>6.2f}{r['sharpe_us']:>7.2f}"
              f"{r['d_sharpe']:>+7.2f}{r['t_d_sharpe']:>+6.2f}   "
              f"{r['corr_underlying']:>9.2f}{r['signal_agree']:>7.0%}   "
              f"{r['alpha_span'] * 100:>+10.2f}{r['t_alpha_span']:>+6.2f}")

    unscored = pairs[(pairs["us_seam"] == "yahoo") & pairs["sharpe_cn"].isna()]
    for _, r in unscored.iterrows():
        print(f"  {r['underlying']:<22}{r['tier']:<7}{int(r['months']):>4}  "
              f"-- under {MIN_MONTHS} shared months, not scored")


def main():
    if "--check" in sys.argv:
        print("=" * 104)
        print("CROSSWALK CHECK")
        print("=" * 104)
        sys.exit(0 if check() else 1)

    OUTDIR.mkdir(exist_ok=True)

    print("=" * 104)
    print("SAME ASSET, TWO MARKETS -- China futures vs their US counterparts")
    print("=" * 104)
    for name in SEAMS:
        built = seam(name)
        print(f"  {built['label']:<20} {built['panel'].shape[1]:>3} instruments   "
              f"{built['panel'].index[0].date()} -> {built['panel'].index[-1].date()}")

    pairs = pair_table()
    print("\n  Sharpe ratios on the months BOTH legs are live. z is the")
    print("  Jobson-Korkie/Memmel test that the two Sharpes differ; corr_ret is")
    print("  the correlation of the raw contracts, agree is 12-month signal")
    print("  agreement; alpha is the China leg's intercept on its US twin.")
    _show_pairs(pairs)

    src = source_control_table()
    scored_src = src[src["d_sharpe_source"].notna()]
    print(f"\n  DATA-SOURCE CONTROL -- Yahoo vs Wind on the same US contract "
          f"({len(scored_src)} underlyings)")
    print(f"    median |Sharpe gap| across feeds : "
          f"{scored_src['d_sharpe_source'].abs().median():.2f}")
    med_market = pairs[(pairs['us_seam'] == 'yahoo')]['d_sharpe'].abs().median()
    print(f"    median |Sharpe gap| across markets: {med_market:.2f}")
    print(f"    median signal agreement, feeds    : "
          f"{scored_src['signal_agree_source'].median():.0%}")
    print(f"    median signal agreement, markets  : "
          f"{pairs[pairs['us_seam'] == 'yahoo']['signal_agree'].median():.0%}")

    classes, series = class_table()
    print("\n  SLEEVE LEVEL -- including AQR's own published factors")
    common = classes[classes["window"] == "common"]
    for cls in ("all", "commodity", "equity", "bond", "currency"):
        block = common[common["asset_class"] == cls]
        if block.empty:
            continue
        n = int(block["months"].iloc[0])
        cells = "  ".join(f"{r['source']:>5} {r['sharpe']:+.2f}"
                          for _, r in block.iterrows())
        print(f"    {cls:<10} (n={n:>3} shared months)  {cells}")

    era = classes[(classes["source"] == "aqr") & (classes["asset_class"] == "all")]
    for _, r in era[era["window"].isin(["paper_era", "post_paper"])].iterrows():
        print(f"    AQR published TSMOM, {r['window']:<11} "
              f"Sharpe {r['sharpe']:+.2f}  ({r['start']} to {r['end']}, n={int(r['months'])})")
    print("    ^ the era control: if AQR's own factor decayed after the paper,")
    print("      China's shortfall is not a China result.")

    pairs.to_csv(OUTDIR / "cross_market_pairs.csv", index=False)
    src.to_csv(OUTDIR / "cross_market_source_control.csv", index=False)
    classes.to_csv(OUTDIR / "cross_market_classes.csv", index=False)
    corrs = class_correlations(series)
    print(f"\n  wrote {OUTDIR / 'cross_market_pairs.csv'}  ({len(pairs)} rows)")
    print(f"  wrote {OUTDIR / 'cross_market_source_control.csv'}  ({len(src)} rows)")
    print(f"  wrote {OUTDIR / 'cross_market_classes.csv'}  ({len(classes)} rows)")
    return pairs, src, classes, corrs


if __name__ == "__main__":
    main()
