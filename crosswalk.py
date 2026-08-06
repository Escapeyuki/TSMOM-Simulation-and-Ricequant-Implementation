"""
Which Chinese futures contract is the same asset as which US one.

WHY THIS FILE EXISTS
    The repo runs the same engine (tsmom.py) through three data seams that never
    talk to each other: data.py (Yahoo, 39 global instruments), data_wind.py
    (Wind, 42), and data_rq.py (Ricequant, 71 China contracts). Their two
    Sharpe-by-instrument figures -- fig2_sharpe_by_instrument_rq.png and the
    right panel of tsmom_replication.png -- report the identical statistic for
    what is very often literally the same metal, the same oilseed, the same
    point on a government yield curve. Nothing in the repo said so.

    replicate_rq.cross_seam_check() hard-codes six such pairs and checks only
    that their raw returns correlate, as a sanity test on the data feed. This
    file generalises that into the actual object of study: a crosswalk that
    lets us ask whether time series momentum travels across markets, and to
    separate two things that a single CN-vs-US comparison confounds --

      market effect       Shanghai gold vs COMEX gold
      data-source effect  COMEX gold via Yahoo vs COMEX gold via Wind

    The second is the control. If Yahoo and Wind disagree about the same US
    contract by as much as China and the US disagree, the cross-market gap is
    not evidence of anything.

KEYED BY (SEAM, CODE), NEVER BY BARE CODE
    The three namespaces collide, and not harmlessly:

        B   -> Soybean No.2   (Ricequant, DCE)   but B.IPE  = Brent   (Wind)
        LC  -> Lithium Carb.  (Ricequant, GFEX)  but LC.CME = Live Cattle (Wind)
        C   -> Corn           (Ricequant, DCE)   and C.CBT  = Corn    (Wind)
        SI  -> Industrial Si  (Ricequant, GFEX)  but SI=F   = Silver  (Yahoo)
        AG  -> Silver         (Ricequant, SHFE)

    The last two are the trap worth naming: China's silver is AG and China's SI
    is industrial silicon, so a mapping built by string similarity gets silver
    exactly wrong. Every lookup here goes through a (seam, code) tuple.

TIERS
    "exact" -- the same physical commodity or the same point on the curve, so a
               Sharpe difference is a statement about the market.
    "proxy" -- economically linked but not the same thing (palm oil against soy
               oil, Shanghai's medium-sour crude against Brent). Kept because
               they are informative, drawn and labelled differently everywhere
               downstream so they are never read as identities.

WHAT IS DELIBERATELY NOT HERE
    Currencies. The paper's fourth sleeve has no liquid domestic Chinese
    futures market, so all seven Yahoo / nine Wind FX contracts are unmatched
    by construction, not by oversight. So are China's ferrous complex, its
    chemicals chain and its shipping contract, none of which have listed US
    equivalents. coverage() reports both sides of that so the gap stays visible.
"""

import pandas as pd

import data
import data_rq
import data_wind

# Sector buckets for the figures. Coarser than the paper's four asset classes
# because within China's commodity sleeve the metals and the oilseeds behave
# like different asset classes, and grouping them together hides that.
SECTORS = ["metals", "energy", "ags", "rates", "equity"]

# (underlying, sector, cn_code, yahoo_ticker, wind_code, tier, note)
#
# wind_code is None where Wind's universe has no counterpart -- it carries no
# cotton and no S&P 500 -- and that is a missing third opinion, not a missing
# pair.
PAIRS = [
    # ---- Metals -----------------------------------------------------------
    ("Gold", "metals", "AU", "GC=F", "GC.CMX", "exact",
     "SHFE gold vs COMEX gold. Both 99.95%+ deliverable bullion; the CNY/USD "
     "move and Shanghai's price limits sit between them."),
    ("Silver", "metals", "AG", "SI=F", "SI.CMX", "exact",
     "SHFE silver vs COMEX silver. Note China's silver is AG -- SI is "
     "industrial silicon, a different commodity entirely."),
    ("Copper", "metals", "CU", "HG=F", "HG.CMX", "exact",
     "SHFE copper vs COMEX copper. The most globally arbitraged of the pairs; "
     "physical flows and the LME sit between the two."),
    ("Copper (bonded)", "metals", "BC", "HG=F", "HG.CMX", "proxy",
     "INE international copper is bonded-warehouse, VAT-free and USD-settled, "
     "so it is a cleaner arbitrage leg than CU but has only traded since 2020."),

    # ---- Energy -----------------------------------------------------------
    ("Crude oil", "energy", "SC", "BZ=F", "B.IPE", "exact",
     "INE crude is a medium-sour Middle East basket priced off Oman/Dubai, so "
     "Brent is the closer benchmark than WTI. Listed 2018."),
    ("Fuel oil", "energy", "FU", "HO=F", "HO.NYM", "proxy",
     "SHFE fuel oil is high-sulphur bunker; NYMEX heating oil is ULSD "
     "distillate. Same barrel, opposite ends of the refinery."),

    # ---- Agriculture ------------------------------------------------------
    ("Soybeans", "ags", "A", "ZS=F", "S.CBT", "exact",
     "DCE Soybean No.1 is non-GM food-grade domestic beans; CBOT is the world "
     "crush benchmark. The two are linked by import parity, not by delivery."),
    ("Soybeans (imported)", "ags", "B", "ZS=F", "S.CBT", "proxy",
     "DCE Soybean No.2 admits GM imported beans, so it tracks CBOT more "
     "directly than A does -- but it was only relaunched with liquidity in 2017."),
    ("Soybean meal", "ags", "M", "ZM=F", "SM.CBT", "exact",
     "DCE meal vs CBOT meal. China crushes imported US and Brazilian beans, so "
     "this is the same product one processing step downstream."),
    ("Soybean oil", "ags", "Y", "ZL=F", "BO.CBT", "exact",
     "DCE soybean oil vs CBOT soybean oil, the other half of the crush."),
    ("Palm oil", "ags", "P", "ZL=F", "BO.CBT", "proxy",
     "DCE palm oil is Malaysian/Indonesian CPO. A substitute for soy oil in "
     "the vegetable-oil complex, not the same oil."),
    ("Rapeseed oil", "ags", "OI", "ZL=F", "BO.CBT", "proxy",
     "CZCE rapeseed oil against soy oil -- again substitution within the "
     "vegetable-oil complex, plus a Canadian canola import channel."),
    ("Rapeseed meal", "ags", "RM", "ZM=F", "SM.CBT", "proxy",
     "CZCE rapeseed meal against soybean meal: competing protein feeds."),
    ("Corn", "ags", "C", "ZC=F", "C.CBT", "exact",
     "DCE corn vs CBOT corn. China's corn is state-reserve managed and tariff "
     "walled, so the two are the same grain in near-segmented markets."),
    ("Sugar", "ags", "SR", "SB=F", "SB.NYB", "exact",
     "CZCE white sugar (refined) vs ICE No.11 raw sugar. Same sweetener, "
     "different refining stage, and China runs an import quota."),
    ("Cotton", "ags", "CF", "CT=F", None, "exact",
     "CZCE cotton vs ICE No.2 cotton. Wind's universe carries no cotton, so "
     "this pair has no data-source control."),
    ("Hogs", "ags", "LH", "HE=F", "LH.CME", "exact",
     "DCE live hog is cash-settled on a national spot average of LIVE hogs; "
     "CME is LEAN hog carcass. Listed 2021, so the shortest ag window here."),

    # ---- Rates ------------------------------------------------------------
    ("2-year government", "rates", "TS", "ZT=F", "TU.CBT", "exact",
     "CFFEX 2-year CGB vs CBOT 2-year Note. Same point on two curves."),
    ("5-year government", "rates", "TF", "ZF=F", "FV.CBT", "exact",
     "CFFEX 5-year CGB vs CBOT 5-year Note."),
    ("10-year government", "rates", "T", "ZN=F", "TY.CBT", "exact",
     "CFFEX 10-year CGB vs CBOT 10-year Note. China's curve is set by the PBoC "
     "and a closed capital account, so co-movement is the open question."),
    ("30-year government", "rates", "TL", "ZB=F", "US.CBT", "exact",
     "CFFEX 30-year CGB vs CBOT Long Bond. TL listed 2023 -- the shortest "
     "window in the whole crosswalk."),

    # ---- Equity -----------------------------------------------------------
    ("Large-cap equity", "equity", "IF", "ES=F", None, "exact",
     "CSI 300 vs S&P 500: each market's own large-cap benchmark future. Wind's "
     "universe has no S&P 500, so no data-source control."),
    ("Mid-cap equity", "equity", "IC", "NQ=F", "ND.CME", "proxy",
     "CSI 500 against the Nasdaq 100 -- matched on being the higher-beta index "
     "rather than on constituents. Wind's Nasdaq series dies in 2015."),
]

# The paper's asset classes, and AQR's published sleeve for each. Used for the
# class-level rollup, where instrument-by-instrument matching is impossible
# (China has no wheat future; the US has no rebar) but the sleeves are still
# comparable objects.
AQR_SLEEVE = {
    "commodity": "TSMOM^CM",
    "equity": "TSMOM^EQ",
    "bond": "TSMOM^FI",
    "currency": "TSMOM^FX",     # US-only: no Chinese counterpart exists
}

# Seam registry. Every downstream file reaches a universe through here rather
# than importing three modules and remembering which is which.
SEAMS = {
    "cn": (data_rq, "Ricequant / China"),
    "yahoo": (data, "Yahoo / global"),
    "wind": (data_wind, "Wind / global"),
}


def universe(seam):
    """The set of instrument codes a seam actually carries."""
    return set(SEAMS[seam][0].ASSET_CLASS)


def name_of(seam, code):
    """Display name for one (seam, code). Falls back to the code itself."""
    return SEAMS[seam][0].INSTRUMENT_NAME.get(code, code)


def asset_class(seam, code):
    """commodity / equity / bond / currency, in the paper's taxonomy."""
    return SEAMS[seam][0].ASSET_CLASS.get(code)


def pairs_frame():
    """PAIRS as a DataFrame, with each leg's display name filled in."""
    rows = []
    for underlying, sector, cn, yahoo, wind, tier, note in PAIRS:
        rows.append({
            "underlying": underlying,
            "sector": sector,
            "tier": tier,
            "asset_class": asset_class("cn", cn),
            "cn": cn, "cn_name": name_of("cn", cn),
            "yahoo": yahoo, "yahoo_name": name_of("yahoo", yahoo) if yahoo else None,
            "wind": wind, "wind_name": name_of("wind", wind) if wind else None,
            "note": note,
        })
    return pd.DataFrame(rows)


def coverage():
    """What the crosswalk reaches and what it leaves behind, per seam.

    Derived from PAIRS rather than hand-listed, so it cannot drift out of date
    when a pair is added. Returns {seam: {"matched": [...], "unmatched": [...]}}
    with display names attached, sorted by asset class then code.
    """
    used = {"cn": set(), "yahoo": set(), "wind": set()}
    for _, _, cn, yahoo, wind, _, _ in PAIRS:
        used["cn"].add(cn)
        if yahoo:
            used["yahoo"].add(yahoo)
        if wind:
            used["wind"].add(wind)

    out = {}
    for seam in SEAMS:
        full = universe(seam)
        def described(codes):
            return sorted(
                ({"code": c, "name": name_of(seam, c), "asset_class": asset_class(seam, c)}
                 for c in codes),
                key=lambda d: (d["asset_class"] or "", d["code"]),
            )
        out[seam] = {
            "matched": described(used[seam] & full),
            "unmatched": described(full - used[seam]),
        }
    return out


def validate():
    """Every code in PAIRS must exist in its seam. Raises on the first that doesn't.

    This is the check that catches a typo'd ticker before it silently becomes a
    dropped row in a figure. Also refuses a duplicate (underlying, tier) key,
    which would double-plot a row.
    """
    problems = []
    for underlying, sector, cn, yahoo, wind, tier, _ in PAIRS:
        if sector not in SECTORS:
            problems.append(f"{underlying}: unknown sector {sector!r}")
        if tier not in ("exact", "proxy"):
            problems.append(f"{underlying}: unknown tier {tier!r}")
        for seam, code in (("cn", cn), ("yahoo", yahoo), ("wind", wind)):
            if code is None:
                continue
            if code not in universe(seam):
                problems.append(f"{underlying}: {code!r} is not in the {seam} universe")

    keys = [p[0] for p in PAIRS]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        problems.append(f"duplicate underlyings: {sorted(dupes)}")

    if problems:
        raise ValueError("crosswalk is inconsistent:\n  " + "\n  ".join(problems))
    return True


def demo():
    """python crosswalk.py -- print the mapping and the coverage it leaves."""
    validate()
    frame = pairs_frame()

    print("=" * 104)
    print("CN <-> US instrument crosswalk")
    print("=" * 104)
    for sector in SECTORS:
        block = frame[frame["sector"] == sector]
        if block.empty:
            continue
        print(f"\n  {sector}")
        for _, r in block.iterrows():
            mark = " " if r["tier"] == "exact" else "~"
            wind = r["wind"] if isinstance(r["wind"], str) else "--"
            print(f"   {mark} {r['underlying']:<21} "
                  f"{r['cn']:>3} {r['cn_name']:<24} <-> "
                  f"{r['yahoo']:<6} {r['yahoo_name']:<20} | wind {wind}")

    n_exact = int((frame["tier"] == "exact").sum())
    print(f"\n  {len(frame)} pairs: {n_exact} exact, {len(frame) - n_exact} proxy (~)")

    print("\n" + "=" * 104)
    print("COVERAGE -- what has no counterpart on the other side")
    print("=" * 104)
    cov = coverage()
    for seam, label in ((s, SEAMS[s][1]) for s in SEAMS):
        m, u = cov[seam]["matched"], cov[seam]["unmatched"]
        print(f"\n  {label}: {len(m)} matched, {len(u)} unmatched of {len(m) + len(u)}")
        by_class = {}
        for d in u:
            by_class.setdefault(d["asset_class"], []).append(d["code"])
        for cls, codes in sorted(by_class.items()):
            print(f"    {cls:<10} ({len(codes):>2}) {' '.join(sorted(codes))}")

    print("\n  The unmatched Chinese commodities are the ferrous complex, the")
    print("  petrochemical chain and the minor ags -- China lists them, the US")
    print("  does not. The unmatched US instruments are led by the currency")
    print("  sleeve, which has no domestic Chinese futures market at all.")
    print("  Those two lists are why the class-level rollup exists.")


if __name__ == "__main__":
    demo()
