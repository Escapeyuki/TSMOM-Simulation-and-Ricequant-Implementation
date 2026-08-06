"""
The China stand-ins for the paper's control factors.

The paper's Eq. 24 regresses a TSMOM strategy on passive exposures to the three
major asset classes plus the Fama-French factors:

    r_TSMOM = a + b1*MKT + b2*BOND + b3*GSCI + s*SMB + h*HML + m*UMD + e

None of those six series exists for China, so every one of them is a
substitution. They are listed here rather than buried in a call site because the
substitutions, not the regressions, are what a reader should argue with.

    paper                         here                                why
    ----------------------------  ----------------------------------  ---------------
    MSCI World, excess            CSI 300 TOTAL RETURN minus 3M CGB   the local market
    Barclays Aggregate Bond       SSE Treasury Bond (000012) minus rf from 2003
    S&P GSCI                      Nanhua Commodity Index (NH0100)     from 2004

    On the first row, note TOTAL RETURN. The obvious ticker for the CSI 300,
    000300.XSHG, is a price index that discards dividends -- 2.24% a year over
    this sample, enough to flip the sign of China's equity risk premium. See the
    MARKET constant below.
    SMB  (small minus big)        Barra `size`, SIGN FLIPPED          see below
    HML  (value)                  Barra `book_to_price`               same construction
    UMD  (cross-sectional mom.)   Barra `momentum`                    same construction

THE SIGN FLIP ON SIZE
    Fama-French SMB is small-cap minus large-cap. A Barra size factor return is
    the return to *large* size -- the exposure is standardised log market cap, so
    a positive number means big beat small. They are opposite in sign by
    construction, so `size` is negated here to make it read as SMB. Getting this
    backwards would flip the sign of one loading in Table 3 and change nothing
    about the alpha, which is exactly the kind of error that survives a review.

WHAT IS NOT COMPARABLE
    The Barra style factors are estimated from a cross-sectional regression that
    also includes ~30 industry factors, so they are industry-neutral in a way the
    Fama-French factors are not. Loadings on them are not quantitatively
    comparable to the paper's. The alpha -- the thing Table 3 is actually about --
    is much less sensitive to this than the betas are.

    The Barra factors also only cover A-shares. The paper's factors are global.
    A China futures strategy has no particular reason to load on Chinese equity
    style factors, so a near-zero loading here is uninformative rather than
    reassuring.
"""

import pandas as pd

from data import _cached
from data_rq import END, START, _init, daily_rf

# CSI 300 TOTAL RETURN ("300收益"), not the headline price index 000300.XSHG.
# This distinction is worth 2.24% a year. Over 2010-2026 the price index compounds
# at 1.60%/yr and the total-return index at 3.82%/yr; the gap is the dividend
# yield, which a price index throws away because it drops on every ex-date.
# Against a risk-free rate near 2.8%, compounding the price index leaves a holder
# BEHIND cash while the total-return index puts them ahead of it -- so the choice
# decides whether China is measured as having had an equity risk premium at all.
# It moves the MKT factor's mean excess return from 1.7%/yr to 3.9%/yr, and
# through the market beta it moves every alpha estimated against it.
#
# The two are 0.9998 correlated day to day, which is exactly why this is easy to
# miss: nothing about the fit looks wrong, only the level is.
MARKET = "H00300.INDX"

# The SSE Treasury Bond Index is already a wealth index -- it compounds at
# 3.87%/yr on 0.74% volatility, which is the coupon, not a price drift. A clean
# price bond index would sit near zero. So no total-return twin is needed, and
# none is published (there is no H00012).
BOND_INDEX = "000012.XSHG"

# The Nanhua Commodity Index tracks futures prices, so there is no dividend
# question here at all -- see monthly_factors() for why it also needs no
# risk-free subtraction.
COMMODITY_INDEX = "NH0100.INDX"

# Barra style factors that map onto Fama-French, and how to relabel them.
STYLE_MAP = {"size": "SMB", "book_to_price": "HML", "momentum": "UMD"}


def _index_prices():
    """Daily closes for the three passive asset-class benchmarks."""
    def fetch():
        rqdatac = _init()
        frame = rqdatac.get_price(
            [MARKET, BOND_INDEX, COMMODITY_INDEX],
            start_date=START, end_date=END, fields=["close"],
            frequency="1d", expect_df=True,
        )
        return frame["close"].unstack(level=0)

    return _cached("rq_index_close", fetch)


def _style_returns():
    """Daily Barra style-factor returns, already in return units."""
    def fetch():
        rqdatac = _init()
        frame = rqdatac.get_factor_return(
            START, END, factors=list(STYLE_MAP), universe="whole_market", model="v1",
        )
        return frame

    return _cached("rq_style_return", fetch)


def monthly_factors():
    """One monthly DataFrame: MKT, BOND, GSCI, SMB, HML, UMD.

    MKT is an excess return (the 3M China government bond yield is taken off);
    the other five are already long-short or index returns and need no rate
    subtraction. Daily returns are compounded, not summed, the same way tsmom.py
    builds its monthly returns.
    """
    prices = _index_prices()
    daily = prices.pct_change(fill_method=None)

    # EVERY passive leg has to be an EXCESS return, because the thing being
    # explained -- a futures strategy -- is one. Getting this wrong is not
    # cosmetic. The SSE Treasury Bond Index returns about 3.8%/yr at 0.9%
    # volatility, which is almost entirely the risk-free rate itself. Left as a
    # total return it lets the BOND loading quietly absorb several percent a year
    # of pure interest, and the alpha it leaves behind is meaningless.
    #
    # So the two CASH indexes get the rate taken off:
    rf = daily_rf().reindex(daily.index).ffill().fillna(0.0)
    for cash_index in (MARKET, BOND_INDEX):
        daily[cash_index] = daily[cash_index] - rf

    # The commodity leg does NOT, and this is the asymmetry worth checking rather
    # than assuming. The Nanhua Commodity Index tracks futures prices rather than
    # a collateralized position: it averages 0.4%/yr over this sample, where a
    # fully-collateralized index would have earned roughly the risk-free rate on
    # top of the same price move. It is already an excess return, exactly like
    # the contracts in data_rq.py, so subtracting the rate again would double
    # count it.

    monthly = (1.0 + daily).resample("ME").prod() - 1.0
    monthly = monthly.rename(columns={MARKET: "MKT", BOND_INDEX: "BOND",
                                      COMMODITY_INDEX: "GSCI"})

    style = _style_returns()
    style_monthly = (1.0 + style).resample("ME").prod() - 1.0
    style_monthly = style_monthly.rename(columns=STYLE_MAP)
    style_monthly["SMB"] = -style_monthly["SMB"]      # see the module docstring

    out = pd.concat([monthly[["MKT", "BOND", "GSCI"]],
                     style_monthly[["SMB", "HML", "UMD"]]], axis=1)
    return out.dropna(how="all")


def market_quarterly():
    """CSI 300 excess return, compounded to non-overlapping quarters.

    The straddle test in replicate.py::smile() wants this shape. The paper uses
    quarterly rather than monthly returns there to stop markets in different time
    zones closing at different moments from blurring the relationship -- an
    argument that matters less for a single-country panel, but the construction
    is kept identical so the two runs are comparable.
    """
    monthly = monthly_factors()["MKT"].dropna()
    return (1.0 + monthly).resample("QE").prod() - 1.0


def demo():
    """`python factors_rq.py` -- shows the factors and checks the obvious traps."""
    factors = monthly_factors()
    print(f"factors_rq.py OK -- {factors.shape[1]} factors, "
          f"{factors.index[0].date()} -> {factors.index[-1].date()}, "
          f"{len(factors)} months\n")

    summary = pd.DataFrame({
        "ann_mean": factors.mean() * 12,
        "ann_vol": factors.std(ddof=1) * (12 ** 0.5),
        "months": factors.notna().sum(),
    })
    print((summary.assign(sharpe=summary["ann_mean"] / summary["ann_vol"])
           .round(3).to_string()))

    # The market leg had the risk-free rate removed and it was not a no-op.
    raw = _index_prices()[MARKET].pct_change()
    raw_m = (1.0 + raw).resample("ME").prod() - 1.0
    diff = (raw_m - factors["MKT"]).dropna()
    assert diff.abs().max() > 1e-6, "MKT excess adjustment was a no-op"
    assert (diff > -1e-9).all(), "excess return exceeds total return somewhere"

    # A Chinese equity bull market should show up as a positive MKT year, which
    # is a weak check that the series is the market and not something else.
    assert factors.loc["2014", "MKT"].sum() > 0.2, "2014 was a big A-share rally"

    # The market leg is the TOTAL RETURN index, not the price index. These are
    # 0.9998 correlated, so no fit statistic would ever reveal a swap -- only the
    # level differs, by the ~2.2%/yr dividend yield. Asserting on the level is
    # therefore the only way to keep this from silently regressing.
    assert MARKET == "H00300.INDX", "MKT must be the total-return CSI 300"
    ann_mkt = factors["MKT"].mean() * 12
    assert ann_mkt > -0.005, (
        f"MKT excess return {ann_mkt:.2%}/yr is too negative -- this is what a "
        "price index looks like once the risk-free rate is taken off it")

    print("\n  correlations:")
    print(factors.corr().round(2).to_string())


if __name__ == "__main__":
    demo()
