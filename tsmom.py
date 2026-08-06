"""
Time Series Momentum (TSMOM) -- the strategy engine.

Paper: Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum",
Journal of Financial Economics 104, 228-250.

WHAT THE PAPER SAYS, IN ONE PARAGRAPH
    Look at how a futures contract did over the last 12 months. If it went up,
    buy it. If it went down, short it. Hold for 1 month, then look again.
    Size every bet so that it risks the same amount (40% annualized volatility),
    then average all the bets together into one portfolio. The paper finds this
    works on all 58 contracts it tests -- commodities, bonds, currencies and
    stock indexes alike.

WHAT THIS FILE IS
    The engine only. You hand it a table of daily *excess* returns (rows = days,
    one column per instrument) and it hands back the TSMOM portfolio. It does
    not know or care where the returns came from, so the same code runs on the
    real data from data.py, wired up by replicate.py. (A synthetic-data demo
    used to live at the bottom of this file; it is commented out now.)

THE FORMULA THIS FILE IMPLEMENTS (paper Eq. 5, Section 4.1):

    r_TSMOM[s, t+1] = sign(past 12-month return of s) * (40% / vol[s, t]) * r[s, t+1]

    ... and then average that across every instrument s available in month t
    to get the single "diversified TSMOM factor".
"""

import numpy as np
import pandas as pd

# ---- Settings, all taken straight from the paper ---------------------------

ANN_DAYS = 261     # Trading days per year. The paper uses 261 in Eq. 1 (Sec. 2.4).
VOL_COM = 60       # "Center of mass" of the volatility average, in days (Sec. 2.4).
VOL_TARGET = 0.40  # Risk 40% annualized per position (Sec. 4.1, Eq. 5).
LOOKBACK_M = 12    # Look back 12 months to decide long vs short (Sec. 4).
MONTHS_PER_YEAR = 12


# ---- Step 1: how risky is each instrument right now? -----------------------

def ex_ante_annual_vol(daily_rets, com=VOL_COM, ann_days=ANN_DAYS):
    """Estimate each instrument's volatility, following the paper's Section 2.4.

    The paper's Eq. 1 is:

        sigma_t^2 = 261 * sum over i of  (1 - d) * d^i * (r[t-1-i] - rbar[t])^2

    In words: take the squared daily returns, weight recent ones more heavily
    than old ones (that is the d^i part, an "exponentially weighted" average),
    then multiply by 261 to turn a daily number into an annual one.

    That weighted average is exactly what pandas' `.ewm(...).var()` computes.
    pandas calls the decay parameter `com` (center of mass); the paper picks
    its `d` so that the center of mass is 60 days, so com=60 here.

    "Ex ante" means we only ever use data from *before* the day we are sizing.
    The paper is explicit about this (end of Sec. 2.4): it uses the volatility
    measured at time t-1 to size the position that earns the time-t return.
    build_tsmom() below does that shifting.
    """
    # bias=True matches the paper's formula, whose weights add up to exactly one.
    #
    # min_periods matters: without it, pandas returns a variance of 0.0 for the
    # very first day of an instrument's history, and 40% / 0 = infinity. The
    # fake data below hides this bug (its first year is dropped anyway), but
    # real data, where instruments start on different dates, does not.
    weighted_variance = daily_rets.ewm(com=com, min_periods=ann_days).var(bias=True)

    # Variance -> volatility, and daily -> annual, in one step.
    return np.sqrt(weighted_variance * ann_days)


# ---- Step 2: the strategy itself -------------------------------------------

def build_tsmom(daily_rets, lookback=LOOKBACK_M, vol_target=VOL_TARGET,
                com=VOL_COM, ann_days=ANN_DAYS):
    """Run the paper's headline strategy: 12-month look-back, 1-month holding.

    Input:  daily_rets -- daily excess returns, rows = dates, columns = instruments.
    Output: (diversified, per_instrument, parts)
              diversified     -- one monthly return series, the TSMOM factor
              per_instrument  -- monthly returns of each instrument's own strategy
              parts           -- the intermediate tables, kept for inspection
    """
    # --- 2a. Volatility, measured daily, then read off at each month end. ---
    daily_vol = ex_ante_annual_vol(daily_rets, com=com, ann_days=ann_days)
    vol_m = daily_vol.resample("ME").last()   # "ME" = month end

    # --- 2b. Turn daily returns into monthly returns. ---
    # Compounding, not adding: 1% then 2% is 1.01 * 1.02 - 1, not 3%.
    monthly_ret = (1.0 + daily_rets).resample("ME").prod() - 1.0

    # --- 2c. The signal: did this instrument go up or down over 12 months? ---
    # Paper Eq. 5 uses sign(r[t-12, t]), the sign of the past 12-month return.
    #
    # We add up 12 monthly log returns instead of compounding 12 simple ones.
    # Both give the same *sign*, which is all we use, and adding is easier to
    # follow than multiplying. log1p(x) is just log(1 + x).
    log_monthly = np.log1p(monthly_ret)
    trailing_12m = log_monthly.rolling(lookback).sum()
    signal = np.sign(trailing_12m)   # +1 = go long, -1 = go short

    # --- 2d. The bet size: 40% / volatility (paper Sec. 4.1). ---
    # A calm bond future gets a big position, a wild natural gas future gets a
    # small one, so that every bet risks roughly the same amount.
    size = vol_target / vol_m

    # Multiply signal by size to get the position, then shift it forward one
    # month. The shift is the whole no-look-ahead rule: the position we hold
    # during month t was decided using only data through the end of month t-1.
    position = (signal * size).shift(1)

    # --- 2e. What each instrument's strategy actually earned. ---
    per_instrument = position * monthly_ret

    # --- 2f. The diversified factor (the line after Eq. 5 in the paper): ---
    # a plain equal-weighted average across the S_t instruments available that
    # month. pandas' .mean() skips missing values, so an instrument with no
    # data that month simply drops out of the average -- exactly the paper's S_t.
    diversified = per_instrument.mean(axis=1)
    diversified.name = "TSMOM"

    parts = {
        "monthly_ret": monthly_ret,
        "vol_m": vol_m,
        "signal": signal,
        "position": position,
    }
    return diversified, per_instrument, parts


# ---- Step 3: scoring a return series ---------------------------------------

def performance(rets, ppy=MONTHS_PER_YEAR):
    """Standard performance stats for a monthly excess-return series.

    `ppy` = periods per year (12 for monthly data), used to annualize.
    """
    r = rets.dropna()

    # Average monthly return -> average annual return.
    ann_mean = r.mean() * ppy

    # Monthly volatility -> annual volatility. Volatility grows with the square
    # root of time, so we multiply by sqrt(12), not by 12.
    ann_vol = r.std(ddof=1) * np.sqrt(ppy)

    # Sharpe ratio = return per unit of risk. These are already excess returns
    # (the risk-free rate is removed), so there is nothing more to subtract.
    # The paper reports a Sharpe above 1 for the diversified factor (Sec. 4.1).
    if ann_vol:
        sharpe = ann_mean / ann_vol
    else:
        sharpe = np.nan

    # Growth of $1 over the whole sample, minus the original $1.
    total_return = (1.0 + r).prod() - 1.0

    # Worst peak-to-trough fall: compare wealth to the highest it had ever been.
    wealth = (1.0 + r).cumprod()
    running_peak = wealth.cummax()
    max_drawdown = (wealth / running_peak - 1.0).min()

    return {
        "months": len(r),
        "ann_mean": ann_mean,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
    }


# =============================================================================
# DATA SEAM -- COMMENTED OUT. Not used by the real calculation.
#
# Everything below is a synthetic (fake) price generator. It was only ever used
# by this file's own main() demo, to smoke-test the engine above without
# downloading anything.
#
# Checked before disabling: make_synthetic_panel, CLASS_SPEC and _ar1_unit are
# referenced nowhere outside this block. replicate.py imports only the engine
# (MONTHS_PER_YEAR, VOL_TARGET, build_tsmom, ex_ante_annual_vol, performance),
# so the real path -- replicate.py -> data.futures_panel() -> build_tsmom() --
# never sees any of it.
#
# To run the fake-data demo again, uncomment from here to the end of the file.
# =============================================================================
#
##
## Everything below here is *not* the paper. It is a synthetic price generator
## used to sanity-check the engine above without downloading anything. The real
## data lives in data.py; replicate.py wires that into build_tsmom().
##
## The instrument counts mirror the paper's Table 1 (24 commodities, 9 equity
## indexes, 13 bonds, 12 currencies = 58) and the volatilities are in the right
## ballpark, so the 40%/vol sizing has something real to do.
#
#CLASS_SPEC = {                  # asset class: (how many, annual volatility)
#    "commodity": (24, 0.30),
#    "equity":    (9,  0.20),
#    "bond":      (13, 0.07),
#    "currency":  (12, 0.11),
#}
#
#
#def _ar1_unit(rng, n, rho):
#    """Generate a slow-moving random wiggle of length n.
#
#    "AR(1)" means each value is mostly a copy of the previous one:
#        x[t] = rho * x[t-1] + a small random shock
#    With rho = 0.98 the series drifts for months at a time instead of jumping
#    around each day. That persistence is what creates trends for TSMOM to find.
#    """
#    x = np.empty(n)
#    x[0] = rng.normal()
#    # The shock size is chosen so the series keeps a variance of about 1.
#    shocks = rng.normal(0, np.sqrt(1 - rho ** 2), n)
#    for t in range(1, n):
#        x[t] = rho * x[t - 1] + shocks[t]
#    return x
#
#
#def make_synthetic_panel(start="1985-01-01", end="2009-12-31",
#                         trend_frac=0.02, rho=0.98, seed=0,
#                         w_global=0.45, w_class=0.20, noise_common=0.05):
#    """Fake daily excess returns for 58 instruments that share common trends.
#
#    Each instrument's return is: a slow drift + fast random noise.
#
#    The drift is deliberately NOT independent across instruments. It is built
#    from a global trend, a per-asset-class trend, and an instrument-specific
#    trend, so the 58 signals move together somewhat -- which is what really
#    happens, and which keeps the diversification benefit realistic instead of
#    pretending 58 independent bets.
#
#    If you set trend_frac to 0 (no drift at all), TSMOM's Sharpe collapses to
#    about zero. That is the engine behaving correctly: no trends, no profits.
#    """
#    rng = np.random.default_rng(seed)
#    dates = pd.bdate_range(start, end)     # business days only
#    n = len(dates)
#
#    # The three drift ingredients. Their weights add up to 1.
#    w_idio = 1.0 - w_global - w_class
#    global_trend = _ar1_unit(rng, n, rho)
#    class_trend = {}
#    for cls in CLASS_SPEC:
#        class_trend[cls] = _ar1_unit(rng, n, rho)
#    common_noise = rng.normal(0, 1, n)     # a shock that hits everything at once
#
#    names = []
#    columns = []
#    for cls, (count, ann_vol) in CLASS_SPEC.items():
#        for j in range(count):
#            # Split each instrument's total daily variance into a small trend
#            # part and a large noise part. Weights are in variance units, so we
#            # take square roots to get standard deviations.
#            daily_vol = ann_vol / np.sqrt(ANN_DAYS)
#            total_var = daily_vol ** 2
#            drift_std = np.sqrt(trend_frac * total_var)
#            noise_std = np.sqrt((1 - trend_frac) * total_var)
#
#            drift = drift_std * (
#                np.sqrt(w_global) * global_trend
#                + np.sqrt(w_class) * class_trend[cls]
#                + np.sqrt(w_idio) * _ar1_unit(rng, n, rho)
#            )
#            noise = noise_std * (
#                np.sqrt(noise_common) * common_noise
#                + np.sqrt(1 - noise_common) * rng.normal(0, 1, n)
#            )
#
#            names.append(f"{cls[:4]}_{j:02d}")
#            columns.append(drift + noise)
#
#    return pd.DataFrame(np.array(columns).T, index=dates, columns=names)
#
#
## ---- Demo ------------------------------------------------------------------
#
#def main():
#    panel = make_synthetic_panel()
#    diversified, per_inst, _ = build_tsmom(panel)
#
#    print(f"Instruments: {panel.shape[1]}   Daily obs: {panel.shape[0]}   "
#          f"Span: {panel.index[0].date()} -> {panel.index[-1].date()}")
#    print("NOTE: numbers below are on SYNTHETIC data -- they validate the "
#          "pipeline, not the paper.\n")
#
#    print("Diversified TSMOM factor:")
#    for name, value in performance(diversified).items():
#        if isinstance(value, float):
#            print(f"  {name:>14}: {value:.4f}")
#        else:
#            print(f"  {name:>14}: {value}")
#
#    # The paper's Fig. 2 shows every one of the 58 contracts with a positive
#    # Sharpe ratio. Here we just count how many of ours are positive.
#    sharpes = per_inst.apply(lambda col: performance(col)["sharpe"])
#    print(f"\nPer-instrument Sharpe: mean={sharpes.mean():.2f}  "
#          f"positive={int((sharpes > 0).sum())}/{len(sharpes)}")
#
#    # Self-check on the sizing rule, and it does not depend on the data at all:
#    # if a position is sized at 40% / vol, then its return should actually show
#    # about 40% annualized volatility. We ignore the long/short sign here,
#    # because flipping sign does not change how volatile something is.
#    monthly_ret = (1.0 + panel).resample("ME").prod() - 1.0
#    size = VOL_TARGET / ex_ante_annual_vol(panel).resample("ME").last()
#    sized_returns = size.shift(1) * monthly_ret
#    realized = (sized_returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)).mean()
#    print(f"\nInvariant check -- mean realized vol per sized position: "
#          f"{realized:.2f} (target {VOL_TARGET:.2f})")
#
#    try:
#        import matplotlib
#        matplotlib.use("Agg")          # write a file, do not open a window
#        import matplotlib.pyplot as plt
#        # Log scale, like the paper's Fig. 3, so equal percentage moves look
#        # equally big whether they happen early or late in the sample.
#        (1 + diversified.dropna()).cumprod().plot(
#            title="Diversified TSMOM — cumulative excess return (synthetic)",
#            figsize=(9, 4), logy=True)
#        plt.tight_layout()
#        plt.savefig("tsmom_equity_curve.png", dpi=110)
#        print("\nSaved equity curve -> tsmom_equity_curve.png")
#    except Exception as e:             # the plot is a nice-to-have, not required
#        print(f"\n(plot skipped: {e})")
#
#
#if __name__ == "__main__":
#    main()
#