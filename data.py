"""
DATA SEAM: builds the real daily excess-return table that tsmom.py consumes.

The paper (Moskowitz, Ooi & Pedersen 2012) uses 58 futures and forward
contracts from January 1965 to December 2009, sourced from Datastream and
Bloomberg (Section 2.1, Appendix A). Those feeds cost money, so this file
builds the closest free substitute from Yahoo Finance. Two consequences:

  - Yahoo's continuous futures start around 2000, so our sample is 2000-present
    instead of the paper's 1985-2009.
  - We get about 34 usable instruments instead of 58.

THE ONE THING THAT IS EASY TO GET WRONG: EXCESS RETURNS
    The paper works entirely in *excess* returns -- returns above the risk-free
    rate. How you get there depends on the instrument:

      - A futures contract is already an excess return. You post margin, not
        the full price, so there is no cash tied up earning interest that needs
        subtracting. Every "=F" ticker below therefore needs no adjustment.
      - A cash stock index (^N225, ^FTSE, ...) is a total return, because you
        would have had to pay for the shares. So we subtract the T-bill rate.

    Using cash indexes in place of stock index futures is something the paper
    itself sanctions: "our return series are almost perfectly correlated with
    the corresponding returns of the underlying cash indexes in excess of the
    Treasury bill rate" (Section 2.1).
"""

import io
import pathlib
import warnings

import numpy as np
import pandas as pd

CACHE = pathlib.Path(__file__).parent / "cache"
PAPER_XLSX = "Time Series Momentum Original Paper Data.xlsx"
UPDATED_XLSX = "Time Series Momentum Factors Monthly.xlsx"

# ---- Which instruments to download -----------------------------------------
# Each row is (ticker, asset class, needs_rf).
#   needs_rf = True  -> a cash index; subtract the risk-free rate.
#   needs_rf = False -> a futures contract; already an excess return.
# The asset classes are the paper's four (Table 1); the counts in the comments
# are what the paper had, for comparison with what Yahoo gives us for free.
INSTRUMENTS = [
    # Commodities (paper: 24) -- Yahoo continuous contracts
    ("CL=F", "commodity", False),  ("BZ=F", "commodity", False),
    ("NG=F", "commodity", False),  ("HO=F", "commodity", False),
    ("RB=F", "commodity", False),  ("GC=F", "commodity", False),
    ("SI=F", "commodity", False),  ("PL=F", "commodity", False),
    ("HG=F", "commodity", False),  ("ZC=F", "commodity", False),
    ("ZS=F", "commodity", False),  ("ZM=F", "commodity", False),
    ("ZL=F", "commodity", False),  ("ZW=F", "commodity", False),
    ("KC=F", "commodity", False),  ("CC=F", "commodity", False),
    ("SB=F", "commodity", False),  ("CT=F", "commodity", False),
    ("LE=F", "commodity", False),  ("HE=F", "commodity", False),
    # Government bonds (paper: 13, across many countries) -- only the US curve
    # is available for free, so we get 4 instead of 13.
    ("ZT=F", "bond", False), ("ZF=F", "bond", False),
    ("ZN=F", "bond", False), ("ZB=F", "bond", False),
    # Currencies (paper: 12) -- CME FX futures, all quoted against the USD
    ("6E=F", "currency", False), ("6J=F", "currency", False),
    ("6B=F", "currency", False), ("6C=F", "currency", False),
    ("6A=F", "currency", False), ("6S=F", "currency", False),
    ("6N=F", "currency", False),
    # Equity indexes (paper: 9) -- US via futures, the rest via cash indexes
    # (hence needs_rf=True on those; see the header note).
    ("ES=F", "equity", False),  ("NQ=F", "equity", False),
    ("^FTSE", "equity", True),  ("^GDAXI", "equity", True),
    ("^N225", "equity", True),  ("^FCHI", "equity", True),
    ("^AEX", "equity", True),   ("^AXJO", "equity", True),
]

# Two lookup tables built from the list above, so other files can ask
# "what class is ES=F?" without scanning.
ASSET_CLASS = {}
NEEDS_RF = {}
for ticker, asset_class, needs_rf in INSTRUMENTS:
    ASSET_CLASS[ticker] = asset_class
    NEEDS_RF[ticker] = needs_rf

# Readable names for charts and printouts. Yahoo's tickers ("6S=F") are not
# something a reader can decode; these are roughly the names the paper uses in
# its own Table 1 and Fig. 2.
INSTRUMENT_NAME = {
    # Commodities
    "CL=F": "Crude Oil (WTI)",   "BZ=F": "Brent Crude",
    "NG=F": "Natural Gas",       "HO=F": "Heating Oil",
    "RB=F": "Gasoline (RBOB)",   "GC=F": "Gold",
    "SI=F": "Silver",            "PL=F": "Platinum",
    "HG=F": "Copper",            "ZC=F": "Corn",
    "ZS=F": "Soybeans",          "ZM=F": "Soybean Meal",
    "ZL=F": "Soybean Oil",       "ZW=F": "Wheat",
    "KC=F": "Coffee",            "CC=F": "Cocoa",
    "SB=F": "Sugar",             "CT=F": "Cotton",
    "LE=F": "Live Cattle",       "HE=F": "Lean Hogs",
    # Government bonds
    "ZT=F": "2-Year US Note",    "ZF=F": "5-Year US Note",
    "ZN=F": "10-Year US Note",   "ZB=F": "30-Year US Bond",
    # Currencies (all against the US dollar)
    "6E=F": "Euro",              "6J=F": "Japanese Yen",
    "6B=F": "British Pound",     "6C=F": "Canadian Dollar",
    "6A=F": "Australian Dollar", "6S=F": "Swiss Franc",
    "6N=F": "NZ Dollar",
    # Equity indexes
    "ES=F": "S&P 500 (US)",      "NQ=F": "Nasdaq 100 (US)",
    "^FTSE": "FTSE 100 (UK)",    "^GDAXI": "DAX (Germany)",
    "^N225": "Nikkei 225 (Japan)", "^FCHI": "CAC 40 (France)",
    "^AEX": "AEX (Netherlands)", "^AXJO": "ASX 200 (Australia)",
}

# A "continuous" futures series is several contracts glued end to end, not one
# real price path. The glue can leave a fake overnight jump, and a back-adjusted
# price can even land near zero (crude oil, April 2020), which makes a
# percentage change explode. Any daily move bigger than this is thrown away
# rather than traded.
MAX_ABS_DAILY_RETURN = 0.40


# ---- Downloading (with a cache, so you only pay for it once) ----------------

def _cached(name, fetch):
    """Return `cache/<name>.csv` if it exists, otherwise call fetch() and save it.

    Delete the file in cache/ if you want fresh data.
    """
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch()
    df.to_csv(path)
    return df


def fetch_prices(start="2000-01-01", end=None):
    """Daily closing prices for every ticker in INSTRUMENTS."""
    def go():
        import yfinance as yf
        tickers = [t for t, _, _ in INSTRUMENTS]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # yfinance is chatty
            raw = yf.download(tickers, start=start, end=end, progress=False,
                              auto_adjust=True)["Close"]
        return raw
    return _cached("prices", go)


def fetch_rf(start="2000-01-01"):
    """The daily risk-free rate: the 3-month US T-bill (FRED series DTB3).

    FRED publishes it as an annual percentage, e.g. 5.25 meaning 5.25% a year.
    We divide by 100 to get a decimal and by 261 to get a per-trading-day rate,
    using the same 261 trading days the paper uses in Section 2.4.
    """
    def go():
        import requests
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
        txt = requests.get(url, timeout=60).text
        d = pd.read_csv(io.StringIO(txt), parse_dates=[0], index_col=0)
        rate = pd.to_numeric(d.iloc[:, 0], errors="coerce").dropna()
        return rate.loc[start:].to_frame("DTB3")
    return _cached("dtb3", go)["DTB3"] / 100.0 / 261.0


# ---- Building the panel -----------------------------------------------------

def futures_panel(start="2000-01-01", end=None):
    """Daily excess returns: one row per date, one column per instrument.

    Each instrument is processed on its own trading calendar and only then
    lined up with the others, which leaves a blank (NaN) wherever an instrument
    did not trade -- Tokyo and Chicago do not share holidays.

    Leaving those blanks is deliberate. Every pandas operation the engine uses
    (.ewm, .resample().prod(), .mean(axis=1)) skips blanks automatically, so a
    missing instrument just drops out of that month's average. That is exactly
    the paper's S_t, "the number of instruments available at time t" (Sec. 4.1).
    Filling the gaps forward instead would invent a 0% day followed by a fake
    jump when trading resumed.
    """
    prices = fetch_prices(start, end)
    rf = fetch_rf(start)

    cols = {}
    for ticker in prices.columns:
        px = prices[ticker].dropna()

        # Guard against the back-adjustment blowup described above.
        px = px[px > 0]

        # Roughly 300 trading days is over a year. Less than that and the
        # 12-month momentum signal never gets a chance to mean anything.
        if len(px) < 300:
            continue

        # Day-over-day percentage change: (today / yesterday) - 1.
        ret = px.pct_change().dropna()
        ret = ret[ret.abs() <= MAX_ABS_DAILY_RETURN]

        # Cash indexes only: turn a total return into an excess return.
        # ffill carries the last published T-bill rate over weekends and
        # holidays; fillna(0) covers any date before FRED's history starts.
        if NEEDS_RF[ticker]:
            daily_rf = rf.reindex(ret.index).ffill().fillna(0.0)
            ret = ret - daily_rf

        cols[ticker] = ret

    panel = pd.concat(cols, axis=1, sort=True)
    return panel.loc[start:]


def load_aqr_factors(which="paper"):
    """Load AQR's published monthly TSMOM factor returns from the Excel files.

    Important: these are the paper's *output*, not its input. They are the
    finished factor the authors built, which we use as a benchmark to check our
    own rebuild against. They are not raw price data.

      which="paper"   -> the original 1985-2009 series from the published paper
      which="updated" -> AQR's version, maintained to the present day

    The two files have different amounts of header junk above the actual table,
    hence the different skiprows.
    """
    root = pathlib.Path(__file__).parent
    if which == "paper":
        fname, skip = PAPER_XLSX, 10
    else:
        fname, skip = UPDATED_XLSX, 17

    df = pd.read_excel(root / fname, skiprows=skip, index_col=0)

    # Rows whose index will not parse as a date are footnotes; drop them.
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]

    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


# ---- Self-check -------------------------------------------------------------

def demo():
    """Run `python data.py` to check the two mistakes that fail silently.

    Both of these would still produce plausible-looking numbers if they were
    wrong, which is exactly why they are worth asserting.
    """
    panel = futures_panel()
    assert panel.shape[1] >= 30, f"only {panel.shape[1]} instruments survived"
    values = panel.to_numpy()
    assert np.isfinite(values[~np.isnan(values)]).all()

    # CHECK 1: cash indexes really did get the risk-free rate subtracted.
    # Rebuild the Nikkei return from scratch and confirm the panel column
    # equals it minus the T-bill -- and that the subtraction was not a no-op.
    raw = fetch_prices()["^N225"].dropna().pct_change().dropna()
    rf = fetch_rf().reindex(raw.index).ffill().fillna(0.0)
    both = pd.concat([panel["^N225"], (raw - rf)], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1]), "^N225 missing rf subtraction"
    assert not np.allclose(both.iloc[:, 0], raw.reindex(both.index)), "rf was a no-op"

    # CHECK 2: no returns were invented by forward-filling prices. Tokyo and
    # Chicago close on different days, so if the panel has zero gaps then
    # something filled them in, and every filled day is a fake 0% return.
    assert panel["^N225"].isna().sum() > 0, "^N225 has no gaps -- returns were ffilled"
    assert (panel.notna().sum(axis=1) < panel.shape[1]).any(), "no staggered coverage"

    print(f"data.py OK -- {panel.shape[1]} instruments, "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}")

    # How many instruments we ended up with in each asset class.
    classes = pd.Series({col: ASSET_CLASS[col] for col in panel.columns})
    print(classes.value_counts().to_string())


if __name__ == "__main__":
    demo()
