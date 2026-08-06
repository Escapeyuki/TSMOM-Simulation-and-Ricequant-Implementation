"""
The TSMOM strategy as an RQAlpha Plus backtest -- what the frictions cost.

replicate_rq.py computes the paper's factor the way the paper does: multiply a
position by a return and average. That is the right way to test the paper's
claim, and it quietly assumes away everything a real futures book has to deal
with -- you cannot hold 3.7 contracts, rolling costs a spread, the exchange wants
margin, and the commission is charged on every one of those trades.

This file runs the identical strategy through the event-driven engine with a
real FUTURE account, and the number worth reading is the GAP between the two
equity curves. That gap is the cost of implementation.

WHERE THE SIGNAL COMES FROM
    The positions are precomputed by tsmom.build_tsmom() and handed to the
    strategy as a table, rather than recomputed bar by bar from history_bars().
    That is deliberate, and it is safe:

      - Safe, because build_tsmom() already does the .shift(1) that makes the
        position held during month t depend only on data through the end of
        month t-1. Nothing in the table is knowable later than the day it is
        used. The look-ahead question is settled in tsmom.py and does not need
        re-litigating here.
      - Deliberate, because the point of this file is to isolate the cost of
        trading. If the signal were recomputed here it could drift from the
        vectorized one -- a slightly different volatility window, a different
        month boundary -- and then the gap between the curves would mix
        implementation cost with signal differences, which is exactly the
        confound the file exists to avoid.

    The one thing this does NOT model is that data_rq.py picks its universe using
    turnover from 2024-2026. A 2011 backtest trading that universe is using a
    list nobody could have written in 2011. It flatters both curves equally, so
    the gap stays meaningful, but the levels are optimistic.

FROM WEIGHTS TO CONTRACTS
    The vectorized book holds a notional weight of (1/S_t) * (40%/vol_i) in each
    instrument. Turning that into lots needs the contract multiplier, and those
    change over time: the futures-mod docs show FB going from 500 to 10 in
    December 2019. Looking the multiplier up per CONTRACT rather than per product
    sidesteps that entirely -- the contracts listed either side of a change carry
    their own correct value, so there is no date to resolve and nothing to get
    wrong. They are read from the backtest bundle via rqalpha's instruments(),
    not over the network -- the bundle's own view is what the bars are priced
    against, and a trial licence has no traffic to spare.

    Rounding to whole lots is itself a friction. On a 10m CNY account the small
    positions round to zero and simply do not get held, which is one of the
    reasons the event-driven curve should sit below the vectorized one.

ROLLS
    The strategy trades the dominant contract. When rqdatac says the dominant
    contract has changed, the old one is closed and the new one opened. The
    paper's continuous series assumes this happens at no cost; here it is charged
    at the going commission and slippage.

Run:
    rqalpha-plus run -f strategy_rq.py -s 2011-03-01 -e 2026-07-31 \
      -fq 1d --account future 10000000 --report ./rq_report
"""

import os
import sys

# rqalpha does not import this file, it reads the source and exec()s it in a
# scope of its own making (rqalpha/utils/strategy_loader_help.py). That scope has
# no __file__ and the project directory is not on sys.path, so the sibling
# imports below fail with ModuleNotFoundError unless the directory is added by
# hand. __file__ is tried first anyway, so that running this file directly --
# for the offline logic tests -- keeps working from any working directory.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # exec'd by rqalpha: no __file__
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from rqalpha.api import instruments, logger, order_to

from data_rq import ASSET_CLASS, futures_panel, load_dominant
from tsmom import build_tsmom

# Enough capital that the smallest sensible position is still more than one lot.
# At 1m CNY most of the book rounds to zero and the backtest measures rounding
# rather than the strategy.
STARTING_CASH = 10_000_000

# Cap on total gross notional as a multiple of net asset value. The paper's own
# footnote 8 says its construction implies 5-20% margin usage, so leverage of a
# few times notional is what the strategy asks for; this is a guard against a
# low-volatility instrument demanding an absurd 40%/vol multiple, not a risk
# model.
MAX_GROSS_LEVERAGE = 4.0


def _month_key(dt):
    return (dt.year, dt.month)


def init(context):
    """Precompute the whole position schedule once, then just trade it."""
    panel = futures_panel()
    _, per_inst, parts = build_tsmom(panel)

    # position[t, s] is the notional weight of instrument s during month t,
    # already lagged by build_tsmom. Divide by the number of live instruments to
    # get the equal-weighted book the paper's S_t average describes.
    breadth = per_inst.notna().sum(axis=1).replace(0, np.nan)
    context.weights = parts["position"].div(breadth, axis=0)

    # Which contract to actually trade on each date.
    context.dominant = load_dominant()

    # Filled lazily from the bundle by _multiplier(); see there for why it does
    # not come from rqdatac.
    context.multipliers = {}
    context.current_month = None
    context.held = {}          # order_book_id -> lots, what we think we hold
    context.rejected = 0       # orders the engine refused to price
    context.rejected_examples = []

    logger.info(f"TSMOM: {panel.shape[1]} instruments, "
                f"{context.weights.index[0].date()} -> {context.weights.index[-1].date()}")


def _multiplier(context, contract):
    """Contract multiplier, read from the backtest bundle and memoised.

    Two reasons this comes from rqalpha's own instruments() rather than from
    rqdatac:

      - It is the bundle's own view of the contract, so the multiplier is
        guaranteed consistent with the bars being priced against it. A second
        opinion fetched over the network could disagree at the edges and would
        silently mis-size positions.
      - It costs no traffic. A trial licence is capped at 1 GB/day and the
        bundle download alone consumes most of it.

    Reading it per CONTRACT rather than per product also removes the date
    question entirely: when a product's multiplier changes -- the futures-mod
    docs show FB going 500 -> 10 in December 2019 -- the contracts listed on
    either side carry their own correct value.
    """
    if contract not in context.multipliers:
        try:
            context.multipliers[contract] = float(
                instruments(contract).contract_multiplier)
        except Exception:
            context.multipliers[contract] = None
    return context.multipliers[contract]


def _tradeable(bar_dict, contract):
    """Can this contract actually be ordered today?

    Used before every order, opening or closing. An unpriceable contract does
    not just get skipped by the engine -- it produces a NaN margin requirement
    and raises, ending the backtest.
    """
    try:
        bar = bar_dict[contract]
    except Exception:
        # The dominant-contract calendar comes from rqdatac and the bars from
        # the bundle; the two can disagree about whether a contract exists on a
        # given date.
        return False
    return bool(bar.is_trading) and not np.isnan(bar.close) and bar.close > 0


def _safe_order(context, contract, lots):
    """Place one order, surviving the ones the engine refuses to price.

    RQAlpha computes an order's frozen cash as
    `calc_cash_occupation(...) + estimated_transaction_cost`, and for a handful
    of historical contracts one of those terms comes back NaN. It then raises
    "Frozen cash of order ... is not supposed to be nan" and the entire backtest
    dies -- in the first full run, at 2015-02-02, four years in.

    The cause is the bundle's own parameter tables, not the strategy: the
    engine warns "trading parameters are abnormal" for pre-2019 contracts,
    because future_info.json ships only with the sample bundle and carries
    current parameters. Prices, margin rates and commission ratios all check out
    individually for the contracts involved, so this is an engine-internal edge
    case that cannot be fixed from strategy code.

    Skipping is also the honest simulation: an order a broker cannot price is an
    order that gets rejected. What matters is that the count stays small, so it
    is tracked and reported at the end of the run rather than swallowed. If it
    were large, the backtest would not be worth quoting.
    """
    try:
        order_to(contract, lots)
        return True
    except Exception as exc:
        context.rejected += 1
        if len(context.rejected_examples) < 5:
            context.rejected_examples.append(
                f"{context.now.date()} {contract} {lots}: {type(exc).__name__}")
        return False


def _target_month(context, date):
    """The most recent month-end row of the weight table at or before `date`.

    The rows are stamped at month end and the position they describe is held
    through the following month, so on any given day we want the last row that
    has already happened.
    """
    rows = context.weights.index[context.weights.index <= pd.Timestamp(date)]
    if len(rows) == 0:
        return None
    return context.weights.loc[rows[-1]]


def _dominant_on(context, date):
    """Each product's dominant contract as of `date`, forward-filled."""
    frame = context.dominant
    rows = frame.index[frame.index <= pd.Timestamp(date)]
    if len(rows) == 0:
        return pd.Series(dtype=object)
    return frame.loc[rows[-1]]


def rebalance(context, bar_dict):
    """Move the book to this month's target. Called on the first bar of a month."""
    date = context.now
    weights = _target_month(context, date)
    if weights is None:
        return
    dominant = _dominant_on(context, date)

    nav = context.portfolio.total_value
    wanted = {}

    for underlying, weight in weights.dropna().items():
        if underlying not in ASSET_CLASS or weight == 0:
            continue
        contract = dominant.get(underlying)
        if not isinstance(contract, str):
            continue

        if not _tradeable(bar_dict, contract):
            continue
        bar = bar_dict[contract]

        mult = _multiplier(context, contract)
        if not mult:
            continue

        # notional wanted / value of one contract = lots
        lots = int(round(nav * weight / (bar.close * mult)))
        if lots != 0:
            wanted[contract] = lots

    # Leverage guard. If the 40%/vol sizing has asked for more gross notional
    # than the account can carry, scale the whole book down rather than dropping
    # instruments -- dropping would change which bets are held, scaling only
    # changes their size. On this panel it binds in 11 months out of 187.
    gross = sum(abs(lots) * bar_dict[c].close * _multiplier(context, c)
                for c, lots in wanted.items())
    if gross > MAX_GROSS_LEVERAGE * nav:
        scale = MAX_GROSS_LEVERAGE * nav / gross
        wanted = {c: int(round(l * scale)) for c, l in wanted.items()}
        wanted = {c: l for c, l in wanted.items() if l != 0}

    # Close anything we hold that is not in the new target -- this is what
    # executes a roll, since a rolled product's old contract simply stops
    # appearing and the new one takes its place.
    #
    # A contract being rolled out of is by definition near the end of its life,
    # so it is the likeliest thing in the book to have stopped trading. Ordering
    # into a contract with no bar at all is what _tradeable() prevents; the
    # separate NaN-frozen-cash failure that _safe_order() catches is an engine
    # edge case unrelated to price (see its docstring). Contracts left unclosed
    # stay in `held` and are retried next month; if the engine has already
    # settled them at expiry, order_to(..., 0) is a harmless no-op.
    for contract in list(context.held):
        if contract in wanted:
            continue
        if _tradeable(bar_dict, contract) and _safe_order(context, contract, 0):
            context.held.pop(contract, None)

    for contract, lots in wanted.items():
        if _safe_order(context, contract, lots):
            context.held[contract] = lots


def after_trading(context):
    """Report rejected orders on the last day of the run.

    Printed rather than silently accumulated, because the whole defence of
    _safe_order() is that the count stays small. A reader has to be able to see
    the number to judge whether the backtest is worth quoting.
    """
    end = getattr(context.run_info, "end_date", None)
    if end is not None and context.now.date() >= end:
        logger.info(f"orders placed without incident; engine refused to price "
                    f"{context.rejected}")
        for line in context.rejected_examples:
            logger.info(f"  e.g. {line}")


def handle_bar(context, bar_dict):
    """Rebalance on the first trading day of each month, hold in between."""
    month = _month_key(context.now)
    if month == context.current_month:
        return
    context.current_month = month
    rebalance(context, bar_dict)


__config__ = {
    "base": {
        "start_date": "2011-03-01",
        "end_date": "2026-07-31",
        "frequency": "1d",
        "accounts": {"future": STARTING_CASH},
    },
    "mod": {
        # Deliberately no output_file / report_save_path here. Setting them makes
        # EVERY run write rq_result.pkl, including short smoke tests, so a stale
        # file from a 10-month run sits where the 15-year result should be and
        # reads as a completed backtest. Pass -o and --report on the command line
        # when you actually want the artifacts.
        "sys_analyser": {"enabled": True},
        "sys_simulation": {
            "enabled": True,
            # Charge the strategy for crossing the spread and for commission at
            # the exchange's own published rates. Without these the event-driven
            # run would just be a slower copy of the vectorized one.
            "matching_type": "current_bar",
            "slippage_model": "PriceRatioSlippage",
            "slippage": 0.0002,
            "futures_commission_multiplier": 1.0,
        },
    },
}
