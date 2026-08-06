"""
DATA SEAM #2: the Wind panel (prices_wind.csv) -> daily excess returns.

data.py builds the same table from Yahoo Finance. This file does it from the
Wind export instead, which is a better fit for the paper in two ways: it starts
in December 1990 rather than 2000, and its futures are real continuous contracts
rather than Yahoo's patchwork.

The excess-return rule is the paper's (Section 2.1), same as data.py:

  - A futures contract is already an excess return -- you post margin, not the
    full notional, so no cash is tied up earning interest. Nothing to subtract.
  - A cash stock index is a total return. Subtract the T-bill.
  - Spot FX is the loose end. The paper uses currency *forwards*, whose excess
    return is roughly the spot move plus the interest-rate differential. We only
    have spot here, so the differential is missing. It is small next to FX
    volatility, and it does not touch the momentum signal, so spot stands in for
    the forward. This is the one approximation in the file.

TWO DATA-QUALITY FILTERS, both needed, both silent failures if skipped:

  1. Wind carries the last close forward through holidays and, worse, through
     the death of a contract. ND.CME (Nasdaq 100) stops moving on 2015-06-18 and
     then repeats one number for eleven more years. Left in, that is 2,700 fake
     0% days: the estimated volatility collapses, 40%/vol explodes, and one dead
     contract swamps the equal-weighted average. Each series is therefore cut at
     the last date its price actually changed.
  2. Within the surviving history, a repeated price is a holiday fill, not a
     flat market. Those days are dropped rather than kept as 0% returns, which
     is the same choice data.py makes when it refuses to forward-fill.
"""

import pathlib

import pandas as pd

from data import _cached

CSV = pathlib.Path(__file__).parent / "prices_wind.csv"

# (Wind code, asset class, needs_rf, display name).
#   needs_rf = True  -> cash index, a total return; subtract the T-bill.
#   needs_rf = False -> futures or spot FX; treated as an excess return already.
INSTRUMENTS = [
    # Commodities (paper: 24)
    ("CL.NYM", "commodity", False, "Crude Oil (WTI)"),
    ("B.IPE", "commodity", False, "Brent Crude"),
    ("NG.NYM", "commodity", False, "Natural Gas"),
    ("HO.NYM", "commodity", False, "Heating Oil"),
    ("RB.NYM", "commodity", False, "Gasoline (RBOB)"),
    ("GC.CMX", "commodity", False, "Gold"),
    ("SI.CMX", "commodity", False, "Silver"),
    ("HG.CMX", "commodity", False, "Copper"),
    ("PL.NYM", "commodity", False, "Platinum"),
    ("C.CBT", "commodity", False, "Corn"),
    ("W.CBT", "commodity", False, "Wheat"),
    ("S.CBT", "commodity", False, "Soybeans"),
    ("SM.CBT", "commodity", False, "Soybean Meal"),
    ("BO.CBT", "commodity", False, "Soybean Oil"),
    ("O.CBT", "commodity", False, "Oats"),
    ("LC.CME", "commodity", False, "Live Cattle"),
    ("LH.CME", "commodity", False, "Lean Hogs"),
    ("CC.NYB", "commodity", False, "Cocoa"),
    ("KC.NYB", "commodity", False, "Coffee"),
    ("SB.NYB", "commodity", False, "Sugar"),
    # Equity indexes (paper: 9). ND.CME is a futures contract; the rest are cash
    # indexes, which is what needs_rf=True is for.
    ("ND.CME", "equity", False, "Nasdaq 100 (US)"),
    ("N225.GI", "equity", True, "Nikkei 225 (Japan)"),
    ("HSI.HI", "equity", True, "Hang Seng (Hong Kong)"),
    ("KOSPI200.KS", "equity", True, "KOSPI 200 (Korea)"),
    ("AS51.GI", "equity", True, "ASX 200 (Australia)"),
    ("SSMI.SIX", "equity", True, "SMI (Switzerland)"),
    ("IBEX.GI", "equity", True, "IBEX 35 (Spain)"),
    ("FTSEMIB.FI", "equity", True, "FTSE MIB (Italy)"),
    # Government bonds (paper: 13)
    ("TU.CBT", "bond", False, "2-Year US Note"),
    ("FV.CBT", "bond", False, "5-Year US Note"),
    ("TY.CBT", "bond", False, "10-Year US Note"),
    ("US.CBT", "bond", False, "30-Year US Bond"),
    ("10JGB.OSE", "bond", False, "10-Year JGB"),
    # Currencies (paper: 12), all spot against the US dollar.
    ("EURUSD.FX", "currency", False, "Euro"),
    ("USDJPY.FX", "currency", False, "Japanese Yen"),
    ("GBPUSD.FX", "currency", False, "British Pound"),
    ("AUDUSD.FX", "currency", False, "Australian Dollar"),
    ("USDCAD.FX", "currency", False, "Canadian Dollar"),
    ("USDCHF.FX", "currency", False, "Swiss Franc"),
    ("NZDUSD.FX", "currency", False, "NZ Dollar"),
    ("USDNOK.FX", "currency", False, "Norwegian Krone"),
    ("USDSEK.FX", "currency", False, "Swedish Krona"),
]

ASSET_CLASS = {}
NEEDS_RF = {}
INSTRUMENT_NAME = {}
for _code, _cls, _rf, _name in INSTRUMENTS:
    ASSET_CLASS[_code] = _cls
    NEEDS_RF[_code] = _rf
    INSTRUMENT_NAME[_code] = _name

# Same guard as data.py: a continuous contract is several contracts glued end to
# end, and the glue can leave a jump that was never tradeable.
MAX_ABS_DAILY_RETURN = 0.40

# Below this share of days actually moving, a series is a stale quote rather
# than a market, and no truncation rescues it.
MIN_MOVING_FRACTION = 0.50


def daily_rf(start="1990-01-01"):
    """Daily risk-free rate from FRED's 3-month T-bill (DTB3).

    Reuses the 1984-onward cache that fig3_cn.py already populated -- data.py's
    own `dtb3` cache is sliced from 2000 and would leave the first decade of the
    Wind sample without a rate. FRED quotes an annual percent (5.25 = 5.25%/yr);
    /100 makes it a decimal and /261 makes it per trading day, using the paper's
    261-day year (Section 2.4).
    """
    rate = _cached("dtb3_1984", lambda: None)["DTB3"]
    return (pd.to_numeric(rate, errors="coerce").dropna() / 100.0 / 261.0).loc[start:]


def load_prices():
    """The raw Wind price table, one column per instrument."""
    px = pd.read_csv(CSV, index_col=0, parse_dates=True)

    # Express every currency as US dollars per unit of foreign currency, so a
    # rising line always means a stronger foreign currency. USDJPY and friends
    # are quoted the other way round, hence the reciprocal.
    for code in px.columns:
        if code.endswith(".FX") and code.startswith("USD"):
            px[code] = 1.0 / px[code]

    return px[[c for c, _, _, _ in INSTRUMENTS if c in px.columns]]


def futures_panel(start=None, end=None):
    """Daily excess returns: one row per date, one column per instrument.

    Instruments keep their own trading calendars and are only lined up at the
    end, so a blank means "did not trade", never "returned 0%". Every operation
    the engine performs (.ewm, .resample().prod(), .mean(axis=1)) skips blanks,
    so a missing instrument simply drops out of that month's average -- the
    paper's S_t, "the number of instruments available at time t" (Section 4.1).
    """
    prices = load_prices()
    rf = daily_rf()

    cols = {}
    for code in prices.columns:
        px = prices[code].dropna()
        px = px[px > 0]
        if len(px) < 300:              # under ~14 months, the 12-month signal never fires
            continue

        # Filter 1: cut the stale tail at the last real price change.
        moved = px.ne(px.shift())
        px = px.loc[:px.index[moved.to_numpy().nonzero()[0][-1]]]

        # Filter 2: drop holiday fills, and refuse a series that is mostly fill.
        moving = px.ne(px.shift())
        moving.iloc[0] = True
        if moving.mean() < MIN_MOVING_FRACTION:
            continue
        px = px[moving]

        ret = px.pct_change().dropna()
        ret = ret[ret.abs() <= MAX_ABS_DAILY_RETURN]

        # Cash indexes only: total return -> excess return. ffill carries the
        # last published T-bill over weekends and holidays.
        if NEEDS_RF[code]:
            ret = ret - rf.reindex(ret.index).ffill().fillna(0.0)

        cols[code] = ret

    panel = pd.concat(cols, axis=1, sort=True)
    return panel.loc[start:end]


def demo():
    """`python data_wind.py` -- checks the things that fail quietly."""
    panel = futures_panel()

    # The staleness filter did something: ND.CME must not reach the end of the
    # file, because it stopped quoting in June 2015.
    nd = panel["ND.CME"].dropna()
    assert nd.index[-1].year <= 2015, "stale ND.CME tail survived the filter"

    # Cash indexes really did have the T-bill removed, and it was not a no-op.
    raw = load_prices()["N225.GI"].dropna()
    raw = raw[raw.ne(raw.shift())].pct_change().dropna()
    rf = daily_rf().reindex(raw.index).ffill().fillna(0.0)
    both = pd.concat([panel["N225.GI"], raw - rf], axis=1).dropna()
    assert (both.iloc[:, 0] - both.iloc[:, 1]).abs().max() < 1e-12, "N225 missing rf"
    assert (both.iloc[:, 0] - raw.reindex(both.index)).abs().max() > 1e-9, "rf was a no-op"

    # Calendars really are staggered. Tokyo and Chicago do not share holidays,
    # so a panel with no gaps would mean something had filled them in.
    assert panel["N225.GI"].isna().sum() > 0, "N225.GI has no gaps -- returns were filled"
    assert (panel.notna().sum(axis=1) < panel.shape[1]).any(), "no staggered coverage"

    print(f"data_wind.py OK -- {panel.shape[1]} instruments, "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}")
    classes = pd.Series({c: ASSET_CLASS[c] for c in panel.columns})
    print(classes.value_counts().to_string())


if __name__ == "__main__":
    demo()
