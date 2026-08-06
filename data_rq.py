"""
DATA SEAM #3: Ricequant (rqdatac) -> daily excess returns for China futures.

data.py builds the same table from Yahoo Finance and data_wind.py from a Wind
export, both of them covering the paper's global markets. This file covers the
one market Ricequant carries: China. Everything downstream is unchanged --
tsmom.py never learns where the returns came from.

WHY FUTURES AND NOT A-SHARES
    The paper is a futures paper. A futures contract is already an *excess*
    return: you post margin, not the full notional, so no cash sits earning
    interest that would have to be subtracted (Section 2.1). That makes the
    whole NEEDS_RF apparatus in data.py and data_wind.py unnecessary here --
    every column below is an excess return as it stands.

HOW THIS UNIVERSE DIFFERS FROM THE PAPER'S 58, and none of it is fixable:

  1. THREE ASSET CLASSES, NOT FOUR. The paper uses commodities, equity indexes,
     bonds and currency forwards. China has no liquid retail FX futures, so the
     currency sleeve is simply absent. The diversified factor averages over
     fewer, more correlated bets than the paper's, so expect a lower Sharpe.
  2. THE SAMPLE STARTS 2010-01-04. Not a choice -- futures.get_dominant_price
     refuses any earlier start date. The paper's window is 1985-2009, so there
     is ZERO overlap. This is an out-of-sample test of the paper's claim on a
     market it never looked at, not a reproduction of its numbers.
  3. THE CLASSES START AT DIFFERENT TIMES. Commodities run the whole sample;
     equity index futures begin with IF in April 2010 (IC/IH 2015, IM 2022);
     bonds begin with TF in September 2013 (T 2015, TS 2018, TL 2023). Early
     months are commodity-only, which is why replicate_rq.py applies a breadth
     filter before reporting anything.

THE ONE THING THAT IS EASY TO GET WRONG HERE: ROLL ADJUSTMENT
    A continuous contract is many contracts glued end to end, and the glue can
    be applied two ways. Ricequant's default, adjust_method='prev_close_spread',
    shifts the whole price level up or down by a *difference* at each roll. That
    is fine for charting and wrong for returns: pct_change() across a shifted
    level is not a return anybody could have earned, and on a long commodity
    history the repeated subtractions can walk the adjusted price to zero or
    negative, at which point the returns are nonsense.

    So this file asks for adjust_method='prev_close_ratio', which glues by
    *ratio*. Then pct_change() is exactly the roll-inclusive return of a held
    futures position -- the paper's r^s_t. This single keyword is the difference
    between a correct panel and a plausible-looking broken one.

QUOTA
    The Ricequant licence in use is a trial: 1 GB of traffic per day. Every
    fetch here is wrapped in data.py's _cached, so the network is touched once
    and then never again. Delete the file in cache/ to force a refresh.
"""

import pandas as pd

from data import _cached

# futures.get_dominant_price raises below this date. It is the floor of the
# continuous-contract service, not a preference.
START = "2010-01-04"

# end_date must be passed explicitly. Left as None the API does NOT run to the
# present -- it returns roughly three months and no warning, which looks like a
# working fetch until you count the rows.
END = pd.Timestamp.today().normalize()

# (symbol, asset class, display name). Produced by screen_universe() below and
# pasted here so that importing this module touches no network, the same way
# data.py and data_wind.py carry static lists. Ordered by recent liquidity.
#
# Contracts listed too recently to have 12 months of history are kept in the
# list rather than pruned by hand -- futures_panel()'s MIN_OBS filter drops them
# automatically, and they enter on their own as history accrues.
INSTRUMENTS = [
    # Equity index futures (paper: 9). All four are CFFEX.
    ("IF", "equity", "CSI 300 Index"),
    ("IH", "equity", "SSE 50 Index"),
    ("IC", "equity", "CSI 500 Index"),
    ("IM", "equity", "CSI 1000 Index"),
    # Government bond futures (paper: 13). All four are CFFEX.
    ("TS", "bond", "2-Year CGB"),
    ("TF", "bond", "5-Year CGB"),
    ("T", "bond", "10-Year CGB"),
    ("TL", "bond", "30-Year CGB"),
    # Commodities (paper: 24). Metals.
    ("AU", "commodity", "Gold"),
    ("AG", "commodity", "Silver"),
    ("CU", "commodity", "Copper"),
    ("AL", "commodity", "Aluminium"),
    ("ZN", "commodity", "Zinc"),
    ("PB", "commodity", "Lead"),
    ("NI", "commodity", "Nickel"),
    ("SN", "commodity", "Tin"),
    ("SS", "commodity", "Stainless Steel"),
    ("BC", "commodity", "International Copper"),
    ("AO", "commodity", "Alumina"),
    ("SI", "commodity", "Industrial Silicon"),
    ("LC", "commodity", "Lithium Carbonate"),
    ("PS", "commodity", "Polysilicon"),
    # Commodities. Ferrous and building materials.
    ("RB", "commodity", "Rebar"),
    ("HC", "commodity", "Hot-Rolled Coil"),
    ("I", "commodity", "Iron Ore"),
    ("J", "commodity", "Coke"),
    ("JM", "commodity", "Coking Coal"),
    ("SF", "commodity", "Ferrosilicon"),
    ("SM", "commodity", "Silicomanganese"),
    ("FG", "commodity", "Glass"),
    ("SA", "commodity", "Soda Ash"),
    # Commodities. Energy and chemicals.
    ("SC", "commodity", "Crude Oil"),
    ("FU", "commodity", "Fuel Oil"),
    ("LU", "commodity", "Low-Sulphur Fuel Oil"),
    ("BU", "commodity", "Bitumen"),
    ("PG", "commodity", "LPG"),
    ("TA", "commodity", "PTA"),
    ("MA", "commodity", "Methanol"),
    ("EG", "commodity", "Ethylene Glycol"),
    ("EB", "commodity", "Styrene"),
    ("PP", "commodity", "Polypropylene"),
    ("L", "commodity", "LLDPE"),
    ("V", "commodity", "PVC"),
    ("UR", "commodity", "Urea"),
    ("PF", "commodity", "Polyester Staple Fibre"),
    ("PX", "commodity", "Paraxylene"),
    ("SH", "commodity", "Caustic Soda"),
    ("BR", "commodity", "Butadiene Rubber"),
    ("PR", "commodity", "Bottle Chip"),
    ("RU", "commodity", "Natural Rubber"),
    ("NR", "commodity", "TSR 20 Rubber"),
    # Commodities. Agriculture.
    ("P", "commodity", "Palm Oil"),
    ("Y", "commodity", "Soybean Oil"),
    ("OI", "commodity", "Rapeseed Oil"),
    ("M", "commodity", "Soybean Meal"),
    ("RM", "commodity", "Rapeseed Meal"),
    ("A", "commodity", "Soybean No.1"),
    ("B", "commodity", "Soybean No.2"),
    ("C", "commodity", "Corn"),
    ("CS", "commodity", "Corn Starch"),
    ("SR", "commodity", "Sugar"),
    ("CF", "commodity", "Cotton"),
    ("CY", "commodity", "Cotton Yarn"),
    ("JD", "commodity", "Eggs"),
    ("LH", "commodity", "Live Hogs"),
    ("AP", "commodity", "Apples"),
    ("CJ", "commodity", "Red Dates"),
    ("PK", "commodity", "Peanut Kernel"),
    ("SP", "commodity", "Wood Pulp"),
    ("LG", "commodity", "Logs"),
    # Commodities. Shipping -- the one non-deliverable contract in the list.
    ("EC", "commodity", "Container Shipping (Europe)"),
]

ASSET_CLASS = {}
INSTRUMENT_NAME = {}
for _code, _cls, _name in INSTRUMENTS:
    ASSET_CLASS[_code] = _cls
    INSTRUMENT_NAME[_code] = _name

# Same guard as the other two seams: a continuous contract is several contracts
# glued end to end, and the glue can leave a jump that was never tradeable.
# Ratio adjustment should make this nearly redundant here; demo() checks that it
# is in fact nearly redundant, because if it starts firing often, the roll
# handling has gone wrong and the guard would hide it.
MAX_ABS_DAILY_RETURN = 0.40

# Below this share of days actually moving, a series is a stale quote rather
# than a market.
MIN_MOVING_FRACTION = 0.50

# Under ~14 months a contract never fires the 12-month signal. Same threshold
# data.py and data_wind.py use.
MIN_OBS = 300

# Liquidity screen: median daily turnover in CNY over the trailing window.
# The paper's universe is explicitly "the 58 liquid instruments we consider".
MIN_TURNOVER = 1e8
LIQ_START = "2024-01-01"

_CLASS_OF_CFFEX = {"IF": "equity", "IH": "equity", "IC": "equity", "IM": "equity",
                   "TS": "bond", "TF": "bond", "T": "bond", "TL": "bond"}

# Zhengzhou renamed several products and Ricequant keeps the old and new codes
# as separate contracts (futures-mod docs, "品种代码切换"). The successors --
# WH, PM, OI, RI, MA, ZC -- carry the tradeable history, so the predecessors are
# dropped. 'S' is the pre-2002 DCE soybean contract, since split into A and B.
# The '_F' codes are alternate listings of L, PP and V.
LEGACY_SYMBOLS = {"ER", "ME", "RO", "TC", "WS", "WT", "S", "L_F", "PP_F", "V_F"}


def _init():
    """Connect to Ricequant. Safe to call repeatedly."""
    import rqdatac
    rqdatac.init()
    return rqdatac


def screen_universe():
    """Rebuild the INSTRUMENTS list above from live data.

    Not called at import time -- its output is pasted into INSTRUMENTS so that
    importing this module is free and the chosen universe is visible in the
    diff. Run `python data_rq.py --screen` to check whether the list has drifted.

    A caveat worth stating out loud: screening on *recent* turnover and then
    running the strategy from 2010 selects instruments we already know survived
    and stayed liquid. The paper does the same thing, so the comparison is fair,
    but the level of the Sharpe ratio is flattered by it either way.
    """
    rqdatac = _init()
    inst = rqdatac.all_instruments(type="Future")
    symbols = sorted(set(inst["underlying_symbol"].dropna()) - LEGACY_SYMBOLS)

    # Turnover must come off the UNADJUSTED series: the adjusted continuous
    # contracts report total_turnover as 0 by construction (futures-mod docs).
    turnover = _cached(
        "rq_turnover",
        lambda: rqdatac.futures.get_dominant_price(
            symbols, start_date=LIQ_START, end_date=END,
            frequency="1d", fields=["total_turnover"], adjust_type="none",
        )["total_turnover"].unstack(level=0),
    )

    median = turnover.median().sort_values(ascending=False)
    liquid = median[median >= MIN_TURNOVER]

    exchange = inst.drop_duplicates("underlying_symbol").set_index("underlying_symbol")["exchange"]
    rows = []
    for symbol in liquid.index:
        cls = _CLASS_OF_CFFEX.get(symbol)
        if cls is None:
            # Everything on CFFEX is index or bond and is named above; anything
            # else on any other exchange is a commodity.
            cls = "commodity" if exchange.get(symbol) != "CFFEX" else "equity"
        rows.append((symbol, cls, INSTRUMENT_NAME.get(symbol, symbol),
                     round(liquid[symbol] / 1e8, 2)))
    return pd.DataFrame(rows, columns=["symbol", "asset_class", "name", "turnover_bn"])


_FRAME = None


def _download():
    """The one network call for prices, memoised for the life of the process.

    Closes and dominant-contract ids arrive in the same response, so fetching
    them separately would spend the trial quota twice for identical bytes.
    """
    global _FRAME
    if _FRAME is None:
        rqdatac = _init()
        symbols = [code for code, _, _ in INSTRUMENTS]
        _FRAME = rqdatac.futures.get_dominant_price(
            symbols, start_date=START, end_date=END, frequency="1d",
            fields=["close"], adjust_type="post", adjust_method="prev_close_ratio",
        )
    return _FRAME


def load_prices():
    """Roll-adjusted daily closes, one column per product, 2010-01-04 onward.

    adjust_method='prev_close_ratio' is the load-bearing argument; see the
    module docstring. adjust_type only decides whether the level is pinned to
    the start or the end of the sample, which pct_change() does not care about.
    """
    return _cached("rq_close", lambda: _download()["close"].unstack(level=0))


def load_dominant():
    """Which contract the continuous series was tracking on each day.

    Not used by the panel, but strategy_rq.py needs it to know when a roll
    happened, and it arrives free with the price fetch.
    """
    return _cached("rq_dominant", lambda: _download()["dominant_id"].unstack(level=0))


def daily_rf(start=START):
    """Daily risk-free rate from the China government bond curve, 3-month tenor.

    NOT applied to the futures panel -- futures returns are already excess
    returns. This exists for the Table 3 work, where the CSI 300 is a cash
    index and does need the rate taken off it.

    get_yield_curve returns an annual decimal (0.0279 = 2.79%/yr); /261 makes it
    per trading day using the paper's 261-day year (Section 2.4).
    """
    def fetch():
        rqdatac = _init()
        curve = rqdatac.get_yield_curve(start_date="2010-01-01", end_date=END, tenor="3M")
        return curve.to_frame("rf") if isinstance(curve, pd.Series) else curve

    curve = _cached("rq_yield_3m", fetch)
    rate = pd.to_numeric(curve.iloc[:, 0], errors="coerce").dropna()
    return (rate / 261.0).loc[start:]


def futures_panel(start=None, end=None):
    """Daily excess returns: one row per date, one column per product.

    Instruments keep their own trading calendars and are only lined up at the
    end, so a blank means "did not trade", never "returned 0%". Every operation
    the engine performs (.ewm, .resample().prod(), .mean(axis=1)) skips blanks,
    so a missing instrument simply drops out of that month's average -- the
    paper's S_t, "the number of instruments available at time t" (Section 4.1).

    No risk-free subtraction anywhere: every column is a futures contract.
    """
    prices = load_prices()

    cols = {}
    for code in prices.columns:
        if code not in ASSET_CLASS:
            continue
        px = prices[code].dropna()
        px = px[px > 0]
        if len(px) < MIN_OBS:
            continue

        # A repeated close is an exchange holiday carried forward, not a market
        # that stood still. Drop those days rather than book them as 0%, which
        # is the same choice the other two seams make.
        moving = px.ne(px.shift())
        moving.iloc[0] = True
        if moving.mean() < MIN_MOVING_FRACTION:
            continue
        px = px[moving]

        ret = px.pct_change().dropna()
        ret = ret[ret.abs() <= MAX_ABS_DAILY_RETURN]
        cols[code] = ret

    panel = pd.concat(cols, axis=1, sort=True)
    return panel.loc[start:end]


def demo():
    """`python data_rq.py` -- checks the things that fail quietly."""
    prices = load_prices()
    panel = futures_panel()

    # Enough of the universe survived to be worth calling a diversified factor.
    assert panel.shape[1] >= 40, f"only {panel.shape[1]} instruments survived"

    # All three asset classes are present, since a single-class factor would
    # not be a replication of anything.
    classes = pd.Series({c: ASSET_CLASS[c] for c in panel.columns})
    counts = classes.value_counts()
    assert set(counts.index) == {"commodity", "equity", "bond"}, counts.to_dict()

    # The contracts really do start when the exchanges say they do. If these
    # ever pass trivially it means the panel got forward-filled backwards.
    assert panel["IF"].dropna().index[0].year == 2010, "IF should start in 2010"
    assert panel["T"].dropna().index[0].year == 2015, "T should start in 2015"
    assert panel["IM"].dropna().index[0].year == 2022, "IM should start in 2022"

    # Calendars are staggered and gaps were left as gaps. A panel with no NaNs
    # would mean something filled them in.
    assert panel["T"].isna().sum() > 0, "T has no gaps -- returns were filled"
    assert (panel.notna().sum(axis=1) < panel.shape[1]).any(), "no staggered coverage"

    # Ratio adjustment should make the 40% return guard nearly redundant. If it
    # is firing often, the roll handling is wrong and the guard is hiding it.
    raw = prices[panel.columns].pct_change()
    clipped = int((raw.abs() > MAX_ABS_DAILY_RETURN).sum().sum())
    assert clipped < 50, f"{clipped} returns above 40% -- check the roll adjustment"

    # No adjusted price walked to zero or negative, the failure mode that spread
    # adjustment would have produced.
    assert (prices.dropna(how="all").min() > 0).all(), "non-positive adjusted price"

    print(f"data_rq.py OK -- {panel.shape[1]} instruments, "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}, "
          f"{clipped} returns clipped")
    print(counts.to_string())
    breadth = panel.notna().sum(axis=1).resample("YE").mean().round(0)
    print("\nmean instruments available per year:")
    print(breadth.to_string())


if __name__ == "__main__":
    import sys
    if "--screen" in sys.argv:
        print(screen_universe().to_string(index=False))
    else:
        demo()
