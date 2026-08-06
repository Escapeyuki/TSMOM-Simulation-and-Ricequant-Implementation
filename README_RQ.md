# Time Series Momentum on China futures, via Ricequant

An out-of-sample test of Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*
(JFE 104, 227–250), on the one market Ricequant covers.

This sits alongside the repo's two existing seams — `data.py` (Yahoo, global) and
`data_wind.py` (Wind, global) — and reuses the same engine. **`tsmom.py` is
unchanged**: it never learns where the returns came from, which is the whole point
of its design.

---

## The headline: the paper's effect does not replicate here

Stated first because it is the result, not a caveat.

| | paper (1985–2009, 58 instruments) | here (2010–2026, 71 instruments) |
|---|---|---|
| Fig. 1 lags 1–12 positive | 12 / 12, nine significant | **8 / 12, one significant** |
| Fig. 2 contracts profitable | 58 / 58, 52 significant | **43 / 65, seven significant** ᵃ |
| Diversified Sharpe | > 1 | **0.62** |
| Factor volatility | ~12% | 13.6% |
| Alpha vs style factors | +1.58%/mo | **+0.39%/mo (t = 1.17)** |
| Straddle: coefficient on mkt² | +1.99 (t = 3.88) | **+0.50 (t = 1.45)** |

ᵃ The panel carries 71 contracts; 65 have the 24 months of strategy history
needed to score one. The other six were listed too recently.

The predictability the paper documents — its core evidence — is largely absent.
What survives is weaker and narrower:

- **TSMOM still beats a passive long book at identical risk.** 8.5%/yr at Sharpe
  0.62 versus 3.4%/yr at Sharpe 0.16, and −25% max drawdown versus −74%. The
  alpha against that benchmark is **+0.74%/mo (t = +2.60)** — the one clearly
  significant result in the whole study.
- **The straddle has the right shape but not the significance.** Negative market
  beta, positive convexity, neither significant. The passive-long control shows
  the *opposite* signs, which is what confirms the test measures what it claims.
- **The 40%/vol sizing behaves exactly as specified** — realized 0.42 against a
  0.40 target — and diversification turns 40% per position into 13.6% for the
  portfolio.

### What the frictions cost

The vectorized factor above is gross of everything. `strategy_rq.py` trades the
identical signal through RQAlpha Plus with a real futures account — whole lots
only, commission on every trade and every roll, slippage, margin — over
2011-03-01 to 2026-07-31 (3,748 trading days, 9,618 trades):

| | ann. return | Sharpe | vol | max DD |
|---|---|---|---|---|
| vectorized, no costs | 8.67% | 0.63 | 13.7% | −24.5% |
| **event-driven, with costs** | **4.43%** | **0.34** | 13.2% | −27.6% |
| passive long, same months | 3.83% | 0.19 | | |

**Implementation costs 4.23% a year and roughly half the Sharpe ratio.** The two
return streams correlate +0.86, which is the check that they are the same
strategy — a low correlation would mean the gap was signal drift rather than
cost. TSMOM still beats the passive long book after costs, but the margin is
thin: 4.43% against 3.83%.

Three caveats, all pushing the same way:

- **The cost is a floor, not an estimate.** `future_info.json` ships only with
  the sample bundle and carries *current* commission rates, so pre-2019 trades
  are charged today's fees. Chinese futures commissions have generally fallen.
- **318 of 9,936 orders (3.2%) were rejected** by an engine bug — RQAlpha
  computes NaN frozen cash for certain historical CZCE contracts and would
  otherwise abort the whole run. They are counted and reported rather than
  swallowed; see `_safe_order()` in `strategy_rq.py`.
- The engine's own summary prints different numbers (3.52%/yr, Sharpe 0.02)
  because it annualizes geometrically and subtracts a risk-free rate from an
  already-excess return. `compare_backtest.py` explains the reconciliation.

### The control that makes this a finding rather than a suspicion

A null result is ambiguous on its own: flat lags could mean China has no time
series momentum, or that `panel_regressions.py` computes the paper's regression
wrongly. Nothing in the China output distinguishes those.

So the identical code is pointed at `data_wind.py`'s 42 **global** futures — the
paper's kind of instrument — and then at their pre-2010 slice, which is as close
to the paper's 1985–2009 window as this repo reaches:

| same code, different data | lags 1–12 positive | significant |
|---|---|---|
| China futures, 2010–2026 | 8 / 12 | 1 |
| Global futures, 1990–2026 | 10 / 12 | 1 |
| **Global futures, 1990–2009** | **9 / 12** | **4** |
| *paper, 1985–2009* | *12 / 12* | *9* |

The pattern strengthens monotonically as the data moves toward the paper's own
universe and era, and the long-lag reversal appears too (28/48 negative in
1990–2009). **The machinery reproduces the paper where the paper's kind of data is
available.** The China result is therefore about China, not about the code.

This is the same argument `replicate.py` makes when it runs the straddle test on
AQR's own factor over AQR's own window before running it on anything else. It
runs automatically as part of `panel_regressions.py`.

### Why "does not replicate" is not the same as "the paper is wrong"

1. **Zero sample overlap.** `futures.get_dominant_price` refuses any start date
   before 2010-01-04. The paper's window is 1985–2009. Not one day in common.
2. **Three asset classes, not four.** China has no liquid retail FX futures, so
   the currency sleeve — a quarter of the paper's diversification — is absent.
3. **Published anomalies decay.** The paper appeared in 2012; this sample is
   almost entirely after publication.
4. **The controls are substitutes, and strong ones.** See below.

---

## Three findings about method that outlived the result

**Clustering the standard errors is the difference between a result and an
artifact.** At lag 1 the pooled panel gives t = **+1.36** clustered by month
versus **+3.48** unclustered. Stacking 70 contracts × 186 months looks like 8,052
independent observations and is nothing of the sort — the contracts move together
within a month. Without the clustering the paper insists on, this study would have
reported a replication.

**Roll adjustment method is load-bearing.** Ricequant's default
`adjust_method='prev_close_spread'` glues contracts by a *difference*, which makes
`pct_change()` something nobody could have earned and can walk a long commodity
history to zero. Switching to `prev_close_ratio` produced **0 returns clipped** by
the 40%-daily-move guard that both other seams need — the cleanest evidence the
panel is right.

**Regressing on high-Sharpe controls destroys alpha whether or not the exposure is
real.** China's size premium runs at Sharpe 1.6 over this window and the bond
premium at 1.7, far above their US counterparts in the paper's era. So
`panel_regressions.py` prints the look-back × holding grid **twice**, with and
without controls. At 12-month look-back / 1-month hold: alpha t = 0.83, raw mean
t = 2.43. The gap is a statement about the controls, not about TSMOM.

**A price index is not a market return.** The obvious CSI 300 ticker,
`000300.XSHG`, is a *price* index: it drops on every ex-dividend date and keeps
none of it. Over 2010–2026 it compounds at 1.60%/yr against the total-return
index's 3.82% — a **2.24%/yr** dividend yield, enough that compounding the price
index leaves a holder behind cash while the total-return index puts them ahead.
The two are 0.9998 correlated day to day, so no fit statistic would ever reveal
the swap; only the level moves. `factors_rq.py` now uses `H00300.INDX` and
asserts on the level, because a correlation check could not catch a regression.

Worth being clear about the scope: this affects only the **control factors**. The
futures panel itself never had a dividend question — see below.

---

## Running it

Everything runs in the **`rq` conda env** (Python 3.11), which is where the
Ricequant stack lives:

```bash
PY=/Users/escape/miniconda3/envs/rq/bin/python

$PY data_rq.py           # seam self-check: 71 instruments, 3 classes, staggered calendars
$PY factors_rq.py        # the six China control factors
$PY replicate_rq.py      # portfolio results, invariants, cross-seam check
$PY panel_regressions.py # Fig. 1 + Table 2   -> fig1_tsmom_predictability_rq.png
$PY table3_rq.py         # alpha + straddle test
$PY fig_rq.py            # Figs. 2 and 3      -> fig2_*.png, fig3_*.png
```

The event-driven backtest needs the data bundle and a live quota:

```bash
rqsdk update-data --base          # once; ~1.3 GB and most of a day's quota
rqalpha-plus run -f strategy_rq.py -s 2011-03-01 -e 2026-07-31 \
  -fq 1d --account future 10000000 -o rq_result.pkl --report ./rq_report
$PY compare_backtest.py           # the cost of implementation
```

Two traps worth knowing before running that, both of which cost real time here
and are written up in `.claude/commands/ricequant-notes.md`:
`download-data --sample` **overwrites** the real bundle with sample-grade data
(back it up first — file size is not a valid check, the sample `futures.h5` is
*larger* while holding half the contracts), and `future_info.json` only ever
comes from `--sample`, while the engine refuses to start without it.

`cache/` holds every network response. Delete a file there to refetch it; leave it
alone and nothing touches the network.

---

## Files

| File | Role |
|---|---|
| `data_rq.py` | The seam. 71 dominant continuous contracts, roll-adjusted by ratio. Same public shape as `data.py` / `data_wind.py`. |
| `factors_rq.py` | China stand-ins for MKT / BOND / GSCI / SMB / HML / UMD. **Read this before believing any alpha.** |
| `replicate_rq.py` | Portfolio performance, the 40%/vol invariant, per-instrument stats, cross-seam sanity check. |
| `panel_regressions.py` | Fig. 1 (Eq. 2 and Eq. 3, SEs clustered by month) and Table 2's grid. |
| `table3_rq.py` | Table 3 Panel A alphas and the Panel C straddle test. |
| `fig_rq.py` | Figs. 2 and 3. |
| `strategy_rq.py` | The same strategy as an RQAlpha Plus futures backtest, to price the frictions. |
| `compare_backtest.py` | The vectorized factor against the traded one — the cost of implementation. |
| `.claude/commands/ricequant-notes.md` | The API traps that cost real time, plus fetch recipes and env notes. Hand-written; `ricequant-doc-index.md` beside it is the generated doc map and is regenerable. |

Only one pre-existing file was modified: `replicate.py` gained an optional
`cluster=` argument on `ols()`, which `panel_regressions.py` needs. `smile()` and
the palette are reused as-is.

---

## Known limitations

- **Universe selection is look-ahead.** `data_rq.py` screens on 2024–2026
  turnover and then trades that list from 2010. The paper screens for liquidity
  too, so the comparison is fair, but the *level* of every Sharpe here is
  flattered. The TSMOM-versus-passive gap is not, since both sides get the same
  universe.
- **The vectorized results are gross of costs.** `compare_backtest.py` prices
  them: 4.23%/yr and about half the Sharpe.
- **Barra style factors are industry-neutral**, unlike Fama-French. Loadings are
  not quantitatively comparable to the paper's; alphas are much less sensitive to
  this than betas.
- **Spot-FX-style approximation is absent here** (no currency sleeve at all),
  unlike `data_wind.py`, which substitutes spot for forwards.
- **13 ghost cells out of 8,143**: `.ewm()` carries its last value across NaN
  rows, so a dormant contract keeps a stale volatility. All are FU (fuel oil,
  dormant 2014–2018). Masking them moves the factor by 5e-5/month, so they are
  left alone rather than special-cased in the shared engine.

## Environment notes

- The Ricequant stack is **only** in the `rq` conda env. A bare `python3` from a
  conda-first shell resolves elsewhere and has no pandas.
- The licence in use is a **trial: 1 GB/day**. `rqsdk update-data --base` consumes
  most of one day on its own, and the RQAlpha Plus engine needs a live rqdatac
  call at startup — so a backtest cannot run at all on an exhausted quota, even
  though the strategy's own data is entirely cached.
