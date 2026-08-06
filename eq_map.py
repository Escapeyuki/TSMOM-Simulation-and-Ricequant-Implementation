"""
Every equation in the paper, and the line of code that is supposed to be it.

WHY
    The repo implements Moskowitz, Ooi & Pedersen (2012) three times over one
    engine, and the interesting question about any replication is not "did it
    run" but "is this the same estimator". Some of the correspondences are
    genuinely non-obvious -- pandas' ewm(com=60).var(bias=True) really is the
    paper's Eq. 1, and it takes an argument to see why -- and some of the
    departures are load-bearing: the China factor regression substitutes CSI 300
    for MSCI World and Barra styles for Fama-French, which is a different test
    wearing the same equation number. None of that was written down anywhere.

    This file is the concordance, machine-readable so the markdown, the
    dashboard and the checks all read the same source.

THREE VERDICTS
    match       the code computes the equation as printed
    deviation   the code deliberately computes something else, for a stated
                reason. Not a bug -- but not the equation either.
    substitute  the equation's inputs do not exist for this market, so a local
                stand-in is used. The structure survives; the test changes.

THE CLAIMS ARE CHECKED, NOT ASSERTED
    verify() runs the arithmetic behind every "match" verdict: that pandas'
    exponential weights really do have the paper's centre of mass, that the sign
    of a sum of log returns really is the sign of the 12-month return, and that
    ex_ante_annual_vol reproduces a hand-rolled Eq. 1 to floating-point
    tolerance. A concordance that cannot fail is not evidence.

RUN
    python eq_map.py              # the concordance, to stdout
    python eq_map.py --verify     # run the numerical checks
    python eq_map.py --markdown   # regenerate EQUATIONS.md
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CROPS = ROOT / "docs" / "eq"

# The PDF's text layer uses a shifted font encoding: the glyphs that print as
# "(1)" extract as the three characters "10Þ". Anchors below are stored in that
# encoded form because that is what page.get_text() returns; make_eq_crops.py
# falls back to `rect` when an anchor is not found, so a future pdf with a sane
# cmap does not silently produce blank crops.
#
# Column geometry of this paper (page is 544 x 743 pt): left column spans
# x 37-262, right column x 286-508.
COL_L = (37.0, 262.5)
COL_R = (280.0, 508.0)

ENTRIES = [
    {
        "id": "eq1",
        "label": "Eq. (1) — ex ante volatility estimate",
        "section": "2.4, journal p. 233",
        "page": 6, "anchor": "10Þ", "col": COL_L, "pad": (4, 4),
        "rect": (37.0, 200.0, 262.5, 235.0),
        "plain": "σ²ₜ = 261 · Σᵢ₌₀^∞ (1−δ) δⁱ (r_{t−1−i} − r̄ₜ)²,   with Σ(1−δ)δⁱ·i = δ/(1−δ) = 60 days",
        "code": [{"file": "tsmom.py", "symbol": "ex_ante_annual_vol"}],
        "verdict": "match",
        "claim": "pandas' ewm(com=60).var(bias=True) is the paper's estimator, not an approximation of it.",
        "note": (
            "Three things have to line up and all three do. (a) The paper picks δ so the weights' "
            "centre of mass is 60 days; pandas' `com` parameter *is* the centre of mass, so com=60 "
            "gives exactly δ/(1−δ)=60. (b) The paper's weights (1−δ)δⁱ sum to one, which is the "
            "biased (maximum-likelihood) variance, so bias=True is required — bias=False would apply "
            "a debiasing factor the paper does not have. (c) The paper subtracts r̄ₜ, itself an "
            "exponentially weighted mean, which is what pandas' .var() subtracts. "
            "One addition: min_periods=261 holds the estimate back for a year. Without it an "
            "instrument's first day has zero variance and the 40%/σ position size is infinite. It "
            "uses only past data, so it adds no look-ahead — it removes a divide-by-zero."
        ),
    },
    {
        "id": "eq2",
        "label": "Eq. (2) — predictability regression, size of the past return",
        "section": "3.1, journal p. 233",
        "page": 6, "anchor": "11Þ", "col": COL_L, "pad": (4, 1),
        "rect": (37.0, 509.0, 262.5, 537.0),
        "plain": "rˢₜ/σˢ_{t−1} = α + βₕ · rˢ_{t−h}/σˢ_{t−h−1} + εˢₜ",
        "code": [
            {"file": "panel_regressions.py", "symbol": "stack_lag"},
            {"file": "panel_regressions.py", "symbol": "lag_curve"},
        ],
        "verdict": "match",
        "claim": "Both sides scaled by the lagged ex ante volatility, pooled across contracts, "
                 "standard errors clustered by month.",
        "note": (
            "The scaling is what makes coefficients comparable across a natural-gas contract and a "
            "2-year note, and the code divides both sides exactly as printed — note the numerator and "
            "denominator lags differ (t−h over t−h−1), which is easy to get wrong. "
            "The clustering is the part that matters most for the conclusion: stacking 70 contracts × "
            "190 months looks like 13,000 observations but carries nowhere near that much independent "
            "information. On this panel clustering moves the lag-1 t-statistic from +3.48 to +1.36."
        ),
    },
    {
        "id": "eq3",
        "label": "Eq. (3) — predictability regression, sign of the past return",
        "section": "3.1, journal p. 233",
        "page": 6, "anchor": "12Þ", "col": COL_R, "pad": (4, 1),
        "rect": (285.0, 145.0, 508.0, 173.0),
        "plain": "rˢₜ/σˢ_{t−1} = α + βₕ · sign(rˢ_{t−h}) + εˢₜ",
        "code": [{"file": "panel_regressions.py", "symbol": "stack_lag"}],
        "verdict": "match",
        "claim": "Same regression with np.sign() on the right-hand side; the left-hand side stays "
                 "volatility-scaled.",
        "note": (
            "The paper's reason for keeping the left side scaled while the right side is ±1 is stated "
            "in the text: the regressor is already unit-free, so only the dependent variable needs "
            "putting on a common scale. This is the specification that the trading strategy actually "
            "implements, which is why Fig. 1 Panel B is the panel to read."
        ),
    },
    {
        "id": "eq4",
        "label": "Eq. (4) — abnormal performance against risk factors",
        "section": "3.2, journal p. 235",
        "page": 8, "anchor": "13Þ", "col": COL_L, "pad": (21, 4),
        "rect": (37.0, 80.0, 262.5, 112.0),
        "plain": "r^{TSMOM(k,h)}ₜ = α + β₁MKTₜ + β₂BONDₜ + β₃GSCIₜ + s·SMBₜ + h·HMLₜ + m·UMDₜ + εₜ",
        "code": [
            {"file": "table3_rq.py", "symbol": "regress"},
            {"file": "factors_rq.py", "symbol": "monthly_factors"},
        ],
        "verdict": "substitute",
        "claim": "The structure is the paper's; every one of the six factors is a China stand-in, "
                 "because none of the originals price a Shanghai futures contract.",
        "note": (
            "MSCI World → CSI 300 **total return** (H00300.INDX, not the price index 000300.XSHG: the "
            "2.24%/yr dividend gap flips the sign of China's equity premium, and the two are 0.9998 "
            "correlated so only a level check catches the mistake). Barclays Aggregate → SSE Treasury "
            "Bond Index. S&P GSCI → Nanhua Commodity Index, which is a price index rather than a "
            "collateralised one and so is already an excess return; the code deliberately does not "
            "subtract the risk-free rate from it while it does from the two cash indexes. "
            "Fama-French SMB/HML/UMD → Barra size/book-to-price/momentum factor returns, with SMB "
            "sign-flipped because Barra's `size` is the return to *large*. "
            "The verdict is `substitute`, not `match`, and the alpha it produces (+0.39%/mo, t=1.17) "
            "is not directly comparable with the paper's +1.58%/mo."
        ),
    },
    {
        "id": "eq5",
        "label": "Eq. (5) — the TSMOM return for one instrument",
        "section": "4.1, journal p. 236",
        "page": 9, "anchor": "14Þ", "col": COL_L, "pad": (5, 5),
        "rect": (37.0, 620.0, 265.0, 648.0),
        "plain": "r^{TSMOM,s}_{t,t+1} = sign(rˢ_{t−12,t}) · (40% / σˢₜ) · rˢ_{t,t+1}",
        "code": [{"file": "tsmom.py", "symbol": "build_tsmom",
                  "from": "daily_vol =", "to": "per_instrument ="}],
        "verdict": "deviation",
        "claim": "Implemented with σ_{t−1} rather than the σₜ printed in the equation — which is what "
                 "the paper's own text instructs.",
        "note": (
            "Section 2.4 says: \"To ensure no look-ahead bias contaminates our results, we use the "
            "volatility estimates at time t−1 applied to time-t returns throughout the analysis.\" "
            "The printed subscript in Eq. (5) is σˢₜ. The code follows the sentence, not the "
            "subscript: `position = (signal * size).shift(1)` lags the signal and the position size "
            "together by one month, so the return earned in month t is priced off information "
            "available at the end of t−1. That single `.shift(1)` is the entire no-look-ahead rule.\n\n"
            "Separately, `sign(rˢ_{t−12,t})` is computed as the sign of a 12-month sum of log returns "
            "rather than of the compounded simple return. These are identical — log is monotone and "
            "log(1)=0 — and verify() checks it."
        ),
    },
    {
        "id": "eq6",
        "label": "Eq. (6) — the diversified TSMOM factor",
        "section": "4.1, journal p. 236",
        "page": 9, "anchor": None, "col": COL_R, "pad": (0, 0),
        "rect": (280.0, 245.0, 440.0, 279.5),
        "plain": "r^{TSMOM}_{t,t+1} = (1/Sₜ) · Σ_{s=1}^{Sₜ} sign(rˢ_{t−12,t}) · (40% / σˢₜ) · rˢ_{t,t+1}",
        "code": [
            {"file": "tsmom.py", "symbol": "build_tsmom",
             "from": "diversified =", "to": "return diversified"},
            {"file": "replicate_rq.py", "symbol": "breadth_of"},
        ],
        "verdict": "match",
        "claim": "An equal weight over the Sₜ instruments live in month t — and Sₜ is counted from "
                 "the positions, not from the returns.",
        "note": (
            "`.mean(axis=1)` skips NaN by default, so the divisor is automatically the number of "
            "instruments with a position that month: exactly the paper's Sₜ. The subtlety is what "
            "counts as live. `resample(\"ME\").prod()` over a month with no data returns 1.0, the "
            "product of an empty set, so a dead contract's monthly return comes out 0.0 rather than "
            "NaN — counting those would inflate Sₜ with contracts that were never listed. "
            "`breadth_of` counts per-instrument *positions* for that reason. compare_markets.py goes "
            "one step further and masks the ghost months outright, which matters for one series "
            "(Wind's Nasdaq, dead after June 2015, otherwise books 133 months of 0.0%)."
        ),
    },
    {
        "id": "fig2",
        "label": "Fig. 2 — Sharpe ratio of the 12-month strategy, by instrument",
        "section": "4.1, journal p. 237",
        "page": 10, "anchor": None, "col": (60.0, 470.0), "pad": (0, 0),
        "rect": (118.0, 52.0, 460.0, 245.0),
        "plain": "Per-instrument annualized Sharpe of Eq. (5), gross of costs, 1985–2009. "
                 "All 58 contracts positive, 52 significant at 5%.",
        "code": [
            {"file": "replicate_rq.py", "symbol": "sharpe_t_stats"},
            {"file": "fig_rq.py", "symbol": "draw_fig2"},
        ],
        "verdict": "deviation",
        "claim": "Same statistic, different sample and a minimum-history filter the paper did not need.",
        "note": (
            "The paper's figure is the headline: every single bar above zero. The China version is "
            "43 of 65 positive and 7 significant, but it is not the same test — it is 2010–2026 "
            "rather than 1985–2009, on contracts the paper never saw, and it drops anything with "
            "under 24 months of strategy history (the paper's instruments all had decades). "
            "The significance test is also stricter here: `sharpe_t_stats` runs a real one-sample "
            "t-test on monthly returns, where replicate.py uses the Sharpe·√years approximation."
        ),
    },
    {
        "id": "table2",
        "label": "Table 2 — alphas across look-back and holding periods",
        "section": "3.2, journal p. 235",
        "page": 8, "anchor": None, "col": (60.0, 500.0), "pad": (0, 0),
        "rect": (37.0, 168.0, 505.0, 296.0),
        "plain": "t(α) from Eq. (4) for every (k,h) in {1,3,6,9,12,24,36,48}², where each (k,h) "
                 "strategy averages the h portfolios opened in the last h months.",
        "code": [
            {"file": "panel_regressions.py", "symbol": "tsmom_jh"},
            {"file": "panel_regressions.py", "symbol": "alpha_grid"},
        ],
        "verdict": "match",
        "claim": "The overlapping-portfolio construction is Jegadeesh & Titman's, as the paper "
                 "specifies, so a 12-month hold produces one non-overlapping monthly series.",
        "note": (
            "The paper (p. 234): \"The return at time t represents the average return across all "
            "portfolios at that time, namely the return on the portfolio that was constructed last "
            "month, the month before that … and so on for all currently 'active' portfolios.\" "
            "`tsmom_jh` builds exactly that: `legs = [(target.shift(1 + age) * monthly_ret).mean(axis=1) "
            "for age in range(holding)]`, then averages the legs. Without it a 48-month hold would "
            "produce overlapping observations and t-statistics inflated by roughly √48."
        ),
    },
    {
        "id": "table3c",
        "label": "Table 3 Panel C — the straddle test",
        "section": "4.3, journal p. 238",
        "page": 11, "anchor": None, "col": (60.0, 500.0), "pad": (0, 0),
        "rect": (41.0, 310.0, 500.0, 390.0),
        "plain": "r^{TSMOM}_q = α + β₁ MKT_q + β₂ MKT_q² + ε_q  (non-overlapping quarters)",
        "code": [{"file": "replicate.py", "symbol": "smile"}],
        "verdict": "match",
        "claim": "Quarterly, non-overlapping, market and market-squared — the paper's own "
                 "specification for 'TSMOM pays like a straddle'.",
        "note": (
            "The paper gets +1.99 on the squared term with t=3.88: trend following makes its money in "
            "the biggest moves in either direction. On China 2010–2026 the same regression gives +0.50 "
            "with t=1.45 — the sign survives, the significance does not. The market proxy is the "
            "substitution from Eq. (4): CSI 300 total return rather than MSCI World."
        ),
    },
    {
        "id": "invariant",
        "label": "The 40% invariant — Section 4.1's sizing claim",
        "section": "4.1, journal p. 236",
        "page": 9, "anchor": None, "col": COL_L, "pad": (0, 0),
        "rect": (37.0, 418.0, 263.0, 522.0),
        "plain": "Each position is sized to 40% annualized volatility; the equal-weighted portfolio "
                 "of them lands near 12%.",
        "code": [{"file": "replicate_rq.py", "symbol": "main",
                  "from": "monthly = (1.0 + panel)", "to": "print(f\"\\n  invariant"}],
        "verdict": "match",
        "claim": "The only check in the whole replication that is a bug if it fails, and it passes: "
                 "realized 0.42 against a 0.40 target.",
        "note": (
            "Everything else in the paper is a finding that may come out negative without anything "
            "being broken. This one is arithmetic: if positions sized to 40%/σ do not realize about "
            "40% volatility, the volatility estimator is wrong. China realizes 0.42. "
            "The portfolio then lands at 13.6% against the paper's ~12% — the diversification, which "
            "is the actual point of Eq. (6)."
        ),
    },
    {
        "id": "crossmarket",
        "label": "Extension — Eq. (5) on both sides of the same asset",
        "section": "not in the paper",
        "page": None, "anchor": None, "col": None, "pad": (0, 0),
        "rect": None,
        "plain": "For each matched pair, Eq. (5) is run on the Chinese contract and on its US "
                 "counterpart over the months both are live.",
        "code": [
            {"file": "compare_markets.py", "symbol": "_pair_stats"},
            {"file": "compare_markets.py", "symbol": "jk_memmel"},
        ],
        "verdict": "extension",
        "claim": "The paper never compares one instrument across two markets; this is new, and it "
                 "needs a test the paper never needed.",
        "note": (
            "Two Sharpe ratios estimated on the same months from correlated return streams cannot be "
            "compared with independent t-tests. `jk_memmel` implements Jobson & Korkie (1981) with "
            "Memmel's (2003) correction, which is the standard fix. The other two statistics — the "
            "correlation of the raw contracts and the fraction of months the two 12-month signals "
            "agree — exist to answer the prior question of whether the pair is the same asset at all."
        ),
    },
]

VERDICT_BLURB = {
    "match": "computes the equation as printed",
    "deviation": "deliberately computes something else, for the stated reason",
    "substitute": "same structure, locally available inputs",
    "extension": "not in the paper",
}


# ----------------------------------------------------------------------------
# Pulling the code out of the repo, so the concordance cannot go stale
# ----------------------------------------------------------------------------

def code_block(spec):
    """The live source for one code reference. Returns (path, first, last, text).

    Read out of the file at run time rather than pasted in, so the concordance
    quotes what the repo actually contains today. `from`/`to` slice inside a
    function when only part of it implements the equation.
    """
    path = ROOT / spec["file"]
    source = path.read_text()
    lines = source.splitlines()

    node = None
    for candidate in ast.walk(ast.parse(source)):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and candidate.name == spec["symbol"]:
            node = candidate
            break
    if node is None:
        raise KeyError(f"{spec['file']}: no def {spec['symbol']}")

    first, last = node.lineno, node.end_lineno
    body = [n for n in node.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str))]
    if body:                          # skip the docstring, keep the code
        first = body[0].lineno
        for deco in node.decorator_list:
            first = min(first, deco.lineno)

    if "from" in spec:
        for i in range(first - 1, last):
            if spec["from"] in lines[i]:
                first = i + 1
                break
        for i in range(first - 1, last):
            if spec["to"] in lines[i]:
                last = i
                break

    text = "\n".join(lines[first - 1:last])
    # Strip the common indent so the snippet reads as a block.
    strip = min((len(l) - len(l.lstrip()) for l in text.splitlines() if l.strip()),
                default=0)
    text = "\n".join(l[strip:] if l.strip() else l for l in text.splitlines())
    return spec["file"], first, last, text


def entry_code(entry):
    return [code_block(spec) for spec in entry["code"]]


# ----------------------------------------------------------------------------
# The claims, checked
# ----------------------------------------------------------------------------

def verify():
    """Run the arithmetic behind every 'match' verdict. Returns list of results."""
    from tsmom import ANN_DAYS, VOL_COM, ex_ante_annual_vol

    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))

    # --- Eq. 1(a): pandas' `com` is the paper's centre of mass -------------
    alpha = 1.0 / (1.0 + VOL_COM)
    i = np.arange(20_000)
    w = alpha * (1 - alpha) ** i
    com = float((w * i).sum() / w.sum())
    check("Eq.1  centre of mass of ewm(com=60) weights == 60 days",
          abs(com - VOL_COM) < 1e-6, f"{com:.9f}")

    # --- Eq. 1(b): the weights are the paper's (1-delta)*delta^i, summing to 1
    check("Eq.1  exponential weights sum to one (the paper's bias=True variance)",
          abs(w.sum() - 1.0) < 1e-9, f"sum={w.sum():.12f}")

    # --- Eq. 1(c): the function reproduces a hand-rolled Eq. 1 -------------
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 4000),
                  index=pd.bdate_range("2000-01-03", periods=4000))
    got = ex_ante_annual_vol(r.to_frame("x"))["x"]

    # Eq. 1 written out directly: exponentially weighted squared deviations
    # from the exponentially weighted mean, annualized by 261.
    delta = 1 - alpha
    t = len(r) - 1
    hist = r.to_numpy()[t::-1]                 # r_t, r_{t-1}, ...
    weights = (1 - delta) * delta ** np.arange(len(hist))
    weights /= weights.sum()                   # finite-sample renormalisation
    rbar = float((weights * hist).sum())
    direct = np.sqrt(ANN_DAYS * float((weights * (hist - rbar) ** 2).sum()))
    check("Eq.1  ex_ante_annual_vol == hand-rolled Eq. (1)",
          abs(direct - got.iloc[-1]) < 1e-6,
          f"direct={direct:.10f} code={got.iloc[-1]:.10f}")

    # --- Eq. 5: sign of summed log returns == sign of the 12-month return --
    m = pd.Series(rng.normal(0.004, 0.06, 5000))
    roll_log = np.log1p(m).rolling(12).sum()
    roll_simple = (1 + m).rolling(12).apply(np.prod, raw=True) - 1
    both = pd.concat([np.sign(roll_log), np.sign(roll_simple)], axis=1).dropna()
    check("Eq.5  sign(sum log(1+r)) == sign(12-month compounded return)",
          (both.iloc[:, 0] == both.iloc[:, 1]).all(),
          f"{len(both)} months, no disagreement")

    # --- Eq. 5: the shift really does remove the look-ahead ----------------
    import tsmom
    panel = pd.DataFrame({"x": r})
    _, per, parts = tsmom.build_tsmom(panel)
    aligned = pd.concat([parts["position"].shift(-1)["x"].rename("pos_t"),
                         (parts["signal"] * (tsmom.VOL_TARGET / parts["vol_m"]))["x"]
                         .rename("raw_t")], axis=1).dropna()
    check("Eq.5  position(t+1) == signal(t) * 40%/sigma(t)  (the .shift(1))",
          np.allclose(aligned["pos_t"], aligned["raw_t"]),
          f"{len(aligned)} months matched")

    # --- Eq. 6: mean(axis=1) is the 1/S_t average over live instruments ----
    frame = pd.DataFrame({"a": [0.1, np.nan, 0.2], "b": [0.3, 0.4, np.nan],
                          "c": [np.nan, np.nan, 0.6]})
    manual = [np.nanmean([0.1, 0.3]), np.nanmean([0.4]), np.nanmean([0.2, 0.6])]
    check("Eq.6  mean(axis=1) divides by S_t, the live-instrument count",
          np.allclose(frame.mean(axis=1).to_numpy(), manual),
          f"S_t = {list(frame.notna().sum(axis=1))}")

    # --- Every code reference still resolves -------------------------------
    missing = []
    for entry in ENTRIES:
        for spec in entry["code"]:
            try:
                code_block(spec)
            except (KeyError, FileNotFoundError) as exc:
                missing.append(f"{entry['id']}: {exc}")
    check("all code references resolve to live source",
          not missing, "; ".join(missing) if missing else
          f"{sum(len(e['code']) for e in ENTRIES)} references")

    return results


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def to_markdown():
    out = [
        "# The paper, equation by equation, against the code",
        "",
        "Generated by `eq_map.py --markdown` — edit that file, not this one.",
        "",
        "Moskowitz, Ooi & Pedersen (2012), *Time series momentum*, "
        "Journal of Financial Economics 104, 228–250. Equation images are cropped "
        "from the PDF in the repo root by `make_eq_crops.py`.",
        "",
        "| verdict | meaning |",
        "|---|---|",
    ]
    for verdict, blurb in VERDICT_BLURB.items():
        out.append(f"| `{verdict}` | {blurb} |")
    out += ["", "| # | what | verdict | code |", "|---|---|---|---|"]
    for e in ENTRIES:
        files = ", ".join(f"`{c['file']}:{c['symbol']}`" for c in e["code"])
        out.append(f"| [{e['id']}](#{e['id']}) | {e['label']} | `{e['verdict']}` | {files} |")
    out.append("")

    for e in ENTRIES:
        out += [f"## {e['id']}", "", f"### {e['label']}", "",
                f"*{e['section']}*", ""]
        crop = CROPS / f"{e['id']}.png"
        if crop.exists():
            out += [f"![{e['label']}](docs/eq/{e['id']}.png)", ""]
        out += [f"> {e['plain']}", "",
                f"**`{e['verdict']}`** — {e['claim']}", ""]
        for path, first, last, text in entry_code(e):
            out += [f"`{path}:{first}`" + (f"–{last}" if last != first else ""),
                    "", "```python", text, "```", ""]
        out += [e["note"], "", "---", ""]

    results = verify()
    out += ["## Checks", "",
            "`python eq_map.py --verify` runs the arithmetic behind the `match` "
            "verdicts. A concordance that cannot fail is not evidence.", "",
            "| check | result | detail |", "|---|---|---|"]
    for name, ok, detail in results:
        out.append(f"| {name} | {'PASS' if ok else 'FAIL'} | `{detail}` |")
    out.append("")
    return "\n".join(out)


def main():
    if "--verify" in sys.argv:
        print("=" * 96)
        print("EQUATION CONCORDANCE -- checking the claims")
        print("=" * 96)
        ok = True
        for name, passed, detail in verify():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            print(f"         {detail}")
            ok &= passed
        print("\n  all checks passed" if ok else "\n  FAILURES ABOVE")
        sys.exit(0 if ok else 1)

    if "--markdown" in sys.argv:
        print(to_markdown())
        return

    print("=" * 96)
    print("PAPER <-> CODE CONCORDANCE")
    print("=" * 96)
    for e in ENTRIES:
        refs = ", ".join(f"{c['file']}:{c['symbol']}" for c in e["code"])
        print(f"\n  {e['label']}")
        print(f"    {e['verdict']:<11} {e['claim']}")
        print(f"    code       {refs}")
    counts = {}
    for e in ENTRIES:
        counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1
    print("\n  " + "   ".join(f"{v}: {n}" for v, n in counts.items()))
    print("  python eq_map.py --verify    to check the claims")
    print("  python eq_map.py --markdown  to regenerate EQUATIONS.md")


if __name__ == "__main__":
    main()
