# Ricequant notes — hand-written

Companion to `ricequant-doc-index.md`. That file is the generated URL map of the
official docs and can be regenerated upstream at any time; **this file cannot**.
Everything below was found by debugging this project and exists nowhere else.

CLAUDE.md requires financial data to come from RQData rather than a web search.
All doc pages are Simplified Chinese, served by VitePress at
`https://www.ricequant.com/doc/`.

## Fetching

The pages render server-side, so `curl` gets the content. WebFetch also works;
do not use WebSearch for these.

```bash
curl -sL "https://www.ricequant.com/doc/rqdata/python/futures-mod"
```

The full page list is machine-readable:

```bash
curl -sL "https://www.ricequant.com/doc/sitemap.xml" | grep -oE '<loc>[^<]+</loc>'
```

Strip the HTML before reading — the nav chrome is most of the bytes:

```bash
python -c "
import re, html, sys
s = sys.stdin.read()
s = re.sub(r'(?s)<script.*?</script>|(?s)<style.*?</style>', '', s)
s = re.sub(r'<[^>]+>', ' ', s)
print(re.sub(r'[ \t]+', ' ', html.unescape(s)))
"
```

## Which page holds which API

`ricequant-doc-index.md` lists every page. This is the short list of the ones
this project actually needs, keyed by the function you are looking for.

Base: `https://www.ricequant.com/doc/`

| Path | Contents |
|---|---|
| `rqdata/python/generic-api` | `all_instruments`, `get_price`, `get_trading_dates`, `get_yield_curve`, `instruments` |
| `rqdata/python/futures-mod` | **futures**: `futures.get_dominant`, `get_dominant_price`, `get_contracts`, `get_ex_factor`, `get_contract_multiplier`, `get_exchange_daily`, `get_continuous_contracts`; continuous-contract rules (88 / 888 / 889 / 99) |
| `rqdata/python/risk-factors-mod` | `get_factor_return`, `get_factor_exposure`, `get_specific_return`, `get_stock_beta` — the Barra-style model |
| `rqdata/python/indices-mod` | index constituents and weights |
| `rqsdk/manual-rqsdk` | `rqsdk license`, `rqsdk install <product>`, `rqsdk update-data`, `rqsdk download-data --sample` |
| `rqalpha-plus/api/config` | accounts, `matching_type`, slippage, `futures_commission_multiplier`, margin |
| `rqalpha-plus/api/order-api` | `order_to`, `order_lots`, `order_target_*`, `buy_open` / `sell_close` / … |
| `rqalpha-plus/api/entrypoint` | `init`, `before_trading`, `handle_bar`, `after_trading` |

## Gotchas found the hard way

These cost real debugging time on this project; check them before assuming a bug.

1. **`get_dominant_price` starts 2010-01-04.** Any earlier `start_date` raises
   `ValueError: expect start_date >= 20100104`. Pre-2010 continuous series must
   be stitched by hand from `futures.get_dominant` plus per-contract `get_price`.
2. **Pass `end_date` explicitly.** With `start_date` set and `end_date=None`,
   `get_dominant_price` returns roughly three months and no warning — it looks
   like a successful fetch until you count the rows.
3. **Use `adjust_method='prev_close_ratio'` when you want returns.** The default,
   `prev_close_spread`, shifts the price level by a *difference* at each roll, so
   `pct_change()` is not a real return and long histories can walk to zero or
   negative. Ratio adjustment makes `pct_change()` the roll-inclusive return.
4. **`total_turnover` is 0 on adjusted continuous contracts** — by construction,
   per the futures-mod docs. Screen liquidity from `adjust_type='none'`.
5. **`dominant_id` is not selectable in daily `fields`** (it is minute/tick only),
   but it is returned anyway in the daily frame.
6. **CZCE renamed products** and Ricequant keeps old and new as separate
   contracts: `WS→WH`, `WT→PM`, `RO→OI`, `ER→RI`, `ME→MA`, `TC→ZC`. The
   successors carry the tradeable history; drop the predecessors.
7. **`contract_multiplier` is time-varying** — the docs show `FB` going 500 → 10
   in December 2019. Query it by date; never cache a scalar.
8. **Any backtest needs `rqsdk update-data --base`** first; the bundle lands in
   `~/.rqalpha-plus/bundle`.
9. **`rqsdk download-data --sample` OVERWRITES the real bundle with sample-grade
   data.** It is not additive, despite being the documented "trial customers"
   path. On this machine it replaced `futures.h5` — 10,884 contracts *with*
   `open_interest` — with a 5,742-contract file *without* it, and the backtest
   then died on `ValueError: Field open_interest does not appear in this type`.
   It rewrote `indexes.h5`, `stocks.h5`, `instruments.pk`, `trading_dates.npy`
   and nine others too. **Back the bundle up before running it**, and note that
   file size is a useless check: the sample `futures.h5` was *larger* on disk
   (145 MB vs 143 MB) while holding half the contracts and one fewer field.
   Compare `len(h5py.File(...).keys())` and `.dtype.names` instead.
10. **`future_info.json` only comes from `--sample`.** `update-data --base` does
    not write it, but the engine refuses to start without it. If `--base` is the
    real source of data, the working combination is: `--base`, then `--sample`,
    then restore every `--base` file from a backup while keeping
    `future_info.json`. Expect "trading parameters are abnormal" warnings for
    historical contracts afterwards — the sample file carries current
    commission/margin parameters only, and the engine falls back to the latest
    values for older dates.

11. **`RuntimeError: Frozen cash of order ... is not supposed to be nan` kills the
    whole backtest**, four years in, with no way to catch it from a mod. RQAlpha
    computes an order's frozen cash as `calc_cash_occupation(...) +
    estimated_transaction_cost` (`rqalpha/portfolio/account.py`), and for some
    historical contracts one term is NaN. Everything checks out individually when
    probed — price, `margin_rate`, `contract_multiplier`, and the commission
    ratios in `future_info.json` are all present and finite, and
    `calc_cash_occupation` returns a sane number — so it is not fixable from
    strategy code. **Every affected contract observed was CZCE** (FG, OI, SR,
    TA), the exchange whose codes Ricequant rewrites. Wrap `order_to` in
    try/except, count the rejections, and report the count: an order the engine
    cannot price is one a broker would reject, but the number has to stay small
    for the run to mean anything.

## Licence and quota

`rqsdk license info` prints the enabled products, the traffic cap and what is
left. A trial licence is capped at 1 GB/day, so cache every fetch to disk — the
seam in `data_rq.py` wraps all of them in `data._cached` for this reason.

## Environment

This project's Ricequant stack lives in the **`rq` conda env**
(`/Users/escape/miniconda3/envs/rq/bin/python`, Python 3.11), not in the
framework Python. `rqdatac`, `rqalpha-plus`, `rqfactor` and `rqoptimizer` are all
installed there and nowhere else.
