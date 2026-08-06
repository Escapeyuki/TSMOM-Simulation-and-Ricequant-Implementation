"""
Run the whole replication once and write every number to one JSON.

WHY
    Every result in this repo lives in a PNG or in stdout. That makes each
    number impossible to check against any other: the Sharpe ratio drawn in
    fig2 and the one printed by replicate_rq are computed by the same code but
    nobody can prove it, and the interactive dashboard would otherwise be a
    fourth independent transcription of the same figures.

    This is the single machine-readable results layer. One run, one file, and
    dashboard.html renders it without recomputing anything.

THE ANCHORS ARE THE POINT
    README_RQ.md publishes specific numbers -- diversified Sharpe 0.62, 43 of 65
    contracts positive, alpha +0.39%/mo, a 4.23%/yr friction cost. If this
    export does not reproduce them exactly then it is wired to something other
    than the code that produced them, and every derived figure is suspect. The
    ANCHORS table at the bottom is checked on every run and a mismatch is a
    non-zero exit.

ONE DELIBERATE INCONSISTENCY, DOCUMENTED
    The `fig2` block uses replicate_rq.sharpe_t_stats unchanged, so the
    dashboard's per-contract tab matches the published PNG bar for bar. The
    `pairs` block uses compare_markets, which additionally masks the months a
    dead contract books a phantom 0.0% return. The mask moves 13 cells of 8,143
    on the China panel; it exists for one series (Wind's Nasdaq, dead after June
    2015) where it is the difference between a number and nonsense.

RUN
    python export_dashboard.py            # -> outputs/dashboard_data.json
    python export_dashboard.py --check    # anchors only, no file written
"""

import base64
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import compare_markets as cmk
import crosswalk
import eq_map
import panel_regressions as pr
import replicate_rq
from data import load_aqr_factors
from replicate import ols
from tsmom import MONTHS_PER_YEAR, VOL_TARGET, performance

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs"
OUT = OUTDIR / "dashboard_data.json"

SEAM_LABEL = {"cn": "China (Ricequant)", "yahoo": "United States (Yahoo)",
              "wind": "United States (Wind)"}
CLASS_ORDER = ["commodity", "equity", "bond", "currency"]


# ----------------------------------------------------------------------------
# JSON hygiene
# ----------------------------------------------------------------------------

def clean(obj):
    """numpy/pandas -> plain JSON, with NaN as null rather than the literal NaN.

    json.dump writes bare NaN by default, which is valid Python and invalid
    JSON: JSON.parse in the browser throws on it.
    """
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else round(value, 8)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, pd.Period, date)):
        return str(obj)
    if isinstance(obj, pd.Series):
        return clean(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return clean(obj.to_dict(orient="records"))
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def series_points(s):
    """A monthly series as [[month, value], ...] with month as 'YYYY-MM'."""
    idx = s.index
    if isinstance(idx, pd.DatetimeIndex):
        idx = idx.to_period("M")
    return [[str(p), clean(v)] for p, v in zip(idx, s.to_numpy())]


# ----------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------

def build_seams():
    """Every seam's engine output, plus the passive-long benchmark."""
    out = {}
    for name in ("cn", "yahoo", "wind"):
        built = cmk.seam(name)
        passive = ((VOL_TARGET / built["parts"]["vol_m"]).shift(1)
                   * built["parts"]["monthly_ret"]).mean(axis=1)
        breadth = built["per_inst"].notna().sum(axis=1)
        keep = breadth[breadth >= replicate_rq.MIN_BREADTH].index
        out[name] = {
            **built,
            "div_filtered": built["per_inst"].mean(axis=1).loc[keep].dropna(),
            "passive_filtered": passive.loc[keep].dropna(),
            "breadth": breadth,
        }
    return out


def scorecard(seams):
    """The headline table: the paper, each seam, and AQR's factor by era."""
    rows = []
    aqr = cmk.as_period(load_aqr_factors(cmk.AQR_WHICH))["TSMOM"].dropna()

    rows.append({
        "source": "paper", "label": "Paper (Moskowitz, Ooi & Pedersen 2012)",
        "window": "1985-01 – 2009-12", "instruments": 58,
        "sharpe": None, "ann_vol": 0.12, "ann_mean": None, "max_drawdown": None,
        "months": 300, "positive": 58, "scored": 58, "significant": 52,
        "alpha_month": 0.0158, "alpha_t": 7.99,
        "note": "Sharpe > 1 (Section 4.1); 58 of 58 instruments positive (Fig. 2); "
                "alpha vs Fama-French +1.58%/mo (Table 3 Panel A).",
    })

    for name, built in seams.items():
        div = built["div_filtered"]
        stats = performance(div)
        tab = replicate_rq.sharpe_t_stats(built["per_inst"])
        # The unfiltered factor too: fig3_wind.py and replicate.py report the
        # global seams without the breadth filter, and on Wind the first year is
        # three contracts, so the two numbers differ by more than rounding.
        unfiltered = performance(built["per_inst"].mean(axis=1).dropna())
        rows.append({
            "source": name, "label": SEAM_LABEL[name],
            "window": f"{div.index[0]:%Y-%m} – {div.index[-1]:%Y-%m}",
            "instruments": int(built["panel"].shape[1]),
            "sharpe": stats["sharpe"], "ann_vol": stats["ann_vol"],
            "sharpe_all_months": unfiltered["sharpe"],
            "months_all": unfiltered["months"],
            "ann_mean": stats["ann_mean"], "max_drawdown": stats["max_drawdown"],
            "months": stats["months"],
            "positive": int((tab["sharpe"] > 0).sum()), "scored": int(len(tab)),
            "significant": int(((tab["t"] > 1.96) & (tab["sharpe"] > 0)).sum()),
            "alpha_month": None, "alpha_t": None,
            "note": f"Diversified factor, months with at least "
                    f"{replicate_rq.MIN_BREADTH} live instruments.",
        })

    for label, window, sub in (
            ("AQR published factor, paper's sample", "1985-01 – 2009-12",
             aqr.loc[:"2009-12"]),
            ("AQR published factor, since", "2010-01 – 2026-05",
             aqr.loc["2010-01":])):
        stats = performance(sub)
        rows.append({
            "source": "aqr", "label": label, "window": window, "instruments": 58,
            "sharpe": stats["sharpe"], "ann_vol": stats["ann_vol"],
            "ann_mean": stats["ann_mean"], "max_drawdown": stats["max_drawdown"],
            "months": stats["months"], "positive": None, "scored": None,
            "significant": None, "alpha_month": None, "alpha_t": None,
            "note": "The authors' own factor, from the AQR data library.",
        })
    return rows


def china_alpha(seams):
    """The China factor's alpha, in both of the paper's specifications.

    Table 3 Panel A is the four Fama-French-style factors (market plus SMB, HML,
    UMD); Eq. (4) adds the bond and commodity indexes. table3_rq.py prints both
    and README_RQ.md quotes the four-factor one (+0.39%/mo, t=1.17), so both are
    exported rather than silently picking one -- they disagree enough to matter
    (+0.39 vs +0.32) and only one of them is the equation.
    """
    import factors_rq
    factors = cmk.as_period(factors_rq.monthly_factors())
    div = cmk.as_period(seams["cn"]["div_filtered"])

    def run(names):
        joined = pd.concat([div.rename("r"), factors[names]], axis=1).dropna()
        table, r2 = ols(joined["r"], [joined[c] for c in names], names)
        return {
            "alpha_month": float(table.loc["intercept", "coef"]),
            "alpha_t": float(table.loc["intercept", "t"]),
            "r2": float(r2), "months": int(len(joined)),
            "loadings": [{"factor": f, "coef": float(table.loc[f, "coef"]),
                          "t": float(table.loc[f, "t"])} for f in names],
        }

    ff4 = run(["MKT", "SMB", "HML", "UMD"])
    return {
        "ff4": ff4,
        "eq4": run(list(factors.columns)),
        # README_RQ.md's headline is the four-factor row; keep it at the top
        # level so the anchor and the dashboard read the same number.
        "alpha_month": ff4["alpha_month"], "alpha_t": ff4["alpha_t"],
    }


def fig1_curves(seams):
    """Paper Fig. 1: pooled predictability at every lag, by spec and class."""
    out = {}
    for name, built in seams.items():
        monthly_ret, vol_lagged = pr.scaled_panels(built["panel"], built["parts"])
        classes = built["module"].ASSET_CLASS
        subsets = {"all": list(monthly_ret.columns)}
        for cls in CLASS_ORDER:
            cols = [c for c in monthly_ret.columns if classes.get(c) == cls]
            if len(cols) >= 2:
                subsets[cls] = cols
        for subset, cols in subsets.items():
            for spec, use_sign in (("size", False), ("sign", True)):
                curve = pr.lag_curve(monthly_ret[cols], vol_lagged[cols],
                                     use_sign=use_sign, max_lag=pr.MAX_LAG)
                out[f"{name}|{subset}|{spec}"] = {
                    "lag": [int(i) for i in curve.index],
                    "t": clean(list(curve["t"])),
                    "coef": clean(list(curve["coef"])),
                    "n": [int(v) for v in curve["n"]],
                }
    return out


def fig2_table(seams):
    """Per-contract Sharpe ratios -- the same numbers the PNGs are drawn from."""
    out = {}
    for name, built in seams.items():
        tab = replicate_rq.sharpe_t_stats(built["per_inst"])
        classes, names = built["module"].ASSET_CLASS, built["module"].INSTRUMENT_NAME
        matched = {p[2] if name == "cn" else (p[3] if name == "yahoo" else p[4])
                   for p in crosswalk.PAIRS}
        out[name] = [{
            "code": code, "name": names.get(code, code),
            "asset_class": classes.get(code),
            "sharpe": clean(row["sharpe"]), "t": clean(row["t"]),
            "months": int(row["months"]),
            "matched": code in matched,
        } for code, row in tab.iterrows()]
    return out


def fig3_curves(seams):
    """Growth of $100: TSMOM against a passive long at the same risk."""
    out = {}
    for name, built in seams.items():
        div, passive = built["div_filtered"], built["passive_filtered"]
        both = pd.concat([cmk.as_period(div).rename("tsmom"),
                          cmk.as_period(passive).rename("passive")],
                         axis=1).dropna()
        both.loc[both.index[0] - 1] = 0.0
        both = both.sort_index()
        wealth = 100 * (1 + both).cumprod()
        out[name] = {
            "months": [str(p) for p in wealth.index],
            "tsmom": clean(list(wealth["tsmom"])),
            "passive": clean(list(wealth["passive"])),
            "stats": {k: clean(performance(both[k].iloc[1:]))
                      for k in ("tsmom", "passive")},
        }
    return out


def raw_grid(built, months=None):
    """Table 2 with no controls: a t-test on each (k,h) strategy's mean return.

    The paper's grid is t(alpha) against six factors. Those factors exist for
    China (factors_rq.py builds local stand-ins) but not for the global seams in
    this repo, so the uncontrolled grid is the one statistic that can be put
    side by side across all three markets.
    """
    monthly_ret, _ = pr.scaled_panels(built["panel"], built["parts"])
    vol_m = built["parts"]["vol_m"]
    grid = {}
    for lookback in pr.PERIODS:
        for holding in pr.PERIODS:
            strategy = pr.tsmom_jh(monthly_ret, vol_m, lookback, holding)
            if months is not None:
                strategy = strategy.reindex(months)
            strategy = strategy.dropna()
            if len(strategy) < 36:
                continue
            t = strategy.mean() / (strategy.std(ddof=1) / np.sqrt(len(strategy)))
            grid[f"{lookback}|{holding}"] = clean(t)
    return grid


def grids(seams):
    import factors_rq
    out = {"periods": pr.PERIODS, "cells": {}}
    monthly_ret, _ = pr.scaled_panels(seams["cn"]["panel"], seams["cn"]["parts"])
    controlled, uncontrolled = pr.alpha_grid(
        monthly_ret, seams["cn"]["parts"]["vol_m"],
        factors_rq.monthly_factors(), months=seams["cn"]["div_filtered"].index)
    for label, frame in (("cn|alpha", controlled), ("cn|raw", uncontrolled)):
        out["cells"][label] = {f"{k}|{c}": clean(frame.loc[k, c])
                               for k in frame.index for c in frame.columns}
    for name in ("yahoo", "wind"):
        out["cells"][f"{name}|raw"] = raw_grid(
            seams[name], months=seams[name]["div_filtered"].index)
    return out


def engine_nav():
    """Monthly returns of the RQAlpha backtest, without needing RQAlpha.

    compare_backtest.py unpickles rq_result.pkl, which only unpickles where
    rqalpha is importable -- the rq conda env, not the environment the rest of
    this analysis runs in. rq_report/portfolio.csv carries the same
    unit_net_value series in plain CSV, so the export reads that and falls back
    to the pickle only if the CSV is missing.
    """
    csv = ROOT / "rq_report" / "portfolio.csv"
    if csv.exists():
        nav = pd.read_csv(csv, index_col=0, parse_dates=True,
                          encoding="utf-8-sig")["unit_net_value"]
        return nav.resample("ME").last().pct_change().dropna(), "rq_report/portfolio.csv"
    from compare_backtest import load_backtest
    return load_backtest()[0], "rq_result.pkl"


def frictions():
    """The vectorized factor against the event-driven backtest."""
    try:
        engine, origin = engine_nav()
    except (SystemExit, FileNotFoundError, ImportError, KeyError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    vector = cmk.as_period(replicate_rq.build()[1])
    both = pd.concat([vector.rename("vectorized"),
                      cmk.as_period(engine).rename("event_driven")],
                     axis=1).dropna()
    stats = {k: clean(performance(both[k])) for k in both.columns}
    trades = ROOT / "rq_report" / "trades.csv"
    return {
        "available": True, "origin": origin,
        "months": [str(p) for p in both.index],
        "vectorized": clean(list(both["vectorized"])),
        "event_driven": clean(list(both["event_driven"])),
        "stats": stats,
        "gap_ann": stats["vectorized"]["ann_mean"] - stats["event_driven"]["ann_mean"],
        "correlation": clean(both["vectorized"].corr(both["event_driven"])),
        "n_trades": sum(1 for _ in trades.open(encoding="utf-8-sig")) - 1
                    if trades.exists() else None,
    }


def equations():
    """The concordance, with each paper crop inlined as a data URI."""
    out = []
    for entry in eq_map.ENTRIES:
        crop = eq_map.CROPS / f"{entry['id']}.png"
        image = None
        if crop.exists():
            image = ("data:image/png;base64,"
                     + base64.b64encode(crop.read_bytes()).decode())
        out.append({
            "id": entry["id"], "label": entry["label"], "section": entry["section"],
            "plain": entry["plain"], "verdict": entry["verdict"],
            "claim": entry["claim"], "note": entry["note"], "image": image,
            "code": [{"file": f, "first": a, "last": b, "text": text}
                     for f, a, b, text in eq_map.entry_code(entry)],
        })
    return {"entries": out, "verdicts": eq_map.VERDICT_BLURB,
            "checks": [{"name": n, "pass": ok, "detail": d}
                       for n, ok, d in eq_map.verify()]}


# ----------------------------------------------------------------------------
# The anchors -- published numbers this export must reproduce
# ----------------------------------------------------------------------------

ANCHORS = [
    ("README: China diversified Sharpe 0.62",
     lambda d: next(r["sharpe"] for r in d["scorecard"] if r["source"] == "cn"),
     0.62, 0.005),
    ("README: China factor volatility 13.6%",
     lambda d: next(r["ann_vol"] for r in d["scorecard"] if r["source"] == "cn"),
     0.136, 0.0005),
    ("README: 43 of 65 China contracts positive",
     lambda d: next(r["positive"] for r in d["scorecard"] if r["source"] == "cn"),
     43, 0),
    ("README: 65 China contracts scoreable",
     lambda d: next(r["scored"] for r in d["scorecard"] if r["source"] == "cn"),
     65, 0),
    ("table3_rq: China alpha vs MKT/SMB/HML/UMD +0.39%/mo",
     lambda d: d["china_alpha"]["ff4"]["alpha_month"] * 100, 0.39, 0.005),
    ("table3_rq: that alpha's t-statistic 1.17",
     lambda d: d["china_alpha"]["ff4"]["alpha_t"], 1.17, 0.005),
    ("table3_rq: China alpha vs all six of Eq. (4) +0.32%/mo",
     lambda d: d["china_alpha"]["eq4"]["alpha_month"] * 100, 0.32, 0.005),
    ("replicate.py: US (Yahoo) diversified Sharpe 0.489",
     lambda d: next(r["sharpe_all_months"] for r in d["scorecard"]
                    if r["source"] == "yahoo"), 0.489, 0.005),
    ("data_wind.py: Wind diversified Sharpe 0.683",
     lambda d: next(r["sharpe_all_months"] for r in d["scorecard"]
                    if r["source"] == "wind"), 0.683, 0.005),
    ("README: friction cost 4.23%/yr",
     lambda d: d["frictions"].get("gap_ann", float("nan")) * 100, 4.23, 0.02),
    ("README: vectorized vs event-driven correlation 0.86",
     lambda d: d["frictions"].get("correlation", float("nan")), 0.86, 0.005),
    ("README: 9,618 trades",
     lambda d: d["frictions"].get("n_trades"), 9618, 0),
]


def check_anchors(data):
    print("\n" + "=" * 96)
    print("ANCHORS -- numbers this export must reproduce or it is wired to the wrong code")
    print("=" * 96)
    ok = True
    for label, getter, expected, tol in ANCHORS:
        try:
            got = getter(data)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [FAIL] {label}\n         raised {exc!r}")
            ok = False
            continue
        passed = got is not None and abs(float(got) - expected) <= tol
        ok &= passed
        shown = "None" if got is None else f"{float(got):.4f}"
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:<58} got {shown}")
    print("\n  all anchors reproduced" if ok else "\n  ANCHOR MISMATCH -- do not trust the export")
    return ok


# ----------------------------------------------------------------------------

def build_all():
    started = time.time()
    seams = build_seams()

    def step(name, fn):
        t0 = time.time()
        value = fn()
        print(f"  {name:<22} {time.time() - t0:6.1f}s")
        return value

    data = {}
    data["meta"] = {
        "generated": str(date.today()),
        "seams": {n: {"label": SEAM_LABEL[n],
                      "instruments": int(b["panel"].shape[1]),
                      "first": str(b["panel"].index[0].date()),
                      "last": str(b["panel"].index[-1].date())}
                  for n, b in seams.items()},
        "vol_target": VOL_TARGET, "months_per_year": MONTHS_PER_YEAR,
        "min_breadth": replicate_rq.MIN_BREADTH, "min_months": cmk.MIN_MONTHS,
    }
    data["scorecard"] = step("scorecard", lambda: scorecard(seams))
    data["china_alpha"] = step("china alpha", lambda: china_alpha(seams))
    data["fig1"] = step("fig1 lag curves", lambda: fig1_curves(seams))
    data["fig2"] = step("fig2 per contract", lambda: fig2_table(seams))
    data["fig3"] = step("fig3 wealth", lambda: fig3_curves(seams))

    pairs = step("cross-market pairs", cmk.pair_table)
    source = step("source control", cmk.source_control_table)
    classes, series = step("class rollups", cmk.class_table)
    data["pairs"] = clean(pairs)
    data["source_control"] = clean(source)
    data["classes"] = clean(classes)
    data["class_curves"] = {
        f"{cls}|{src}": series_points(100 * (1 + s).cumprod())
        for (cls, src), s in series.items()}
    data["crosswalk"] = clean(crosswalk.pairs_frame())
    data["coverage"] = crosswalk.coverage()

    data["grid"] = step("strategy grid", lambda: grids(seams))
    data["frictions"] = step("frictions", frictions)
    data["equations"] = step("equations", equations)

    print(f"  {'total':<22} {time.time() - started:6.1f}s")
    return data


DASHBOARD = ROOT / "dashboard.html"
OPEN_TAG = '<script id="tsmom-data" type="application/json">'
CLOSE_TAG = "</script>"


def splice_into_dashboard(payload):
    """Put the data inside dashboard.html, between its two sentinel tags.

    The published page cannot fetch anything -- a strict content policy blocks
    every request to another host, and a file:// page cannot read a sibling file
    either -- so the data has to live in the document. dashboard.html is both the
    template and the output: only the block between the sentinels is rewritten,
    so the markup and the code in it survive every re-export.
    """
    if not DASHBOARD.exists():
        return None
    html = DASHBOARD.read_text()
    start = html.index(OPEN_TAG) + len(OPEN_TAG)
    end = html.index(CLOSE_TAG, start)
    DASHBOARD.write_text(html[:start] + payload + html[end:])
    return DASHBOARD.stat().st_size


def main():
    data = build_all()
    ok = check_anchors(data)

    if "--check" not in sys.argv:
        OUTDIR.mkdir(exist_ok=True)
        # Escape "<" so the payload can never terminate the script tag it is
        # embedded in. < is valid JSON, so the standalone file still parses.
        payload = json.dumps(clean(data), separators=(",", ":")).replace("<", "\\u003c")
        OUT.write_text(payload)
        print(f"\n  wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")
        size = splice_into_dashboard(payload)
        if size:
            print(f"  wrote {DASHBOARD}  ({size / 1e6:.2f} MB, data inlined)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
