# Coral Reef Survey: Pre-Positioning Sample Kits across a Dive Supply Tree

## Story

A marine institute surveys a coral reef every season. Before an expedition it must
**pre-position single-use sample kits** (vials + specimen tags) at each of `N` dive
**sites**, and it may also hold a shared reserve of kits aboard the research
**vessel** — a central depot that can be ferried out to whichever site runs short on
survey day. The number of specimens actually encountered at a site is **uncertain**;
its distribution (mean, spread, shape) is known from prior seasons, but the exact
count is not. Pre-positioning too many kits wastes budget and cold-storage space
(**holding cost**); pre-positioning too few means specimens go unrecorded
(**shortage cost**); ferrying from the vessel incurs a **transfer cost**; and each
site must meet a **service-level** target (the probability that its own on-site kits
suffice). Choose the local stock at every site and the vessel reserve to **minimize
the total composite cost**.

This is a two-echelon safety-stock / newsvendor problem: local kits = local safety
stock, vessel kits = pooled central safety stock, over-provisioning = holding,
missed specimens = shortage, ferrying = transfer, plus a per-site service constraint.

## You write a program (isolated)

Your program reads **one** public instance as JSON on `stdin` and writes **one** JSON
answer on `stdout`. It runs in an isolated sandbox and only ever sees the public
instance below — the survey-day scenarios used to grade you are hidden in the judge.

### Public instance (stdin)

```json
{
  "name": "reef101",
  "n_sites": 6,
  "sites": [
    {"mean": 41.3, "std": 15.2, "dist": "normal",    "h": 1.8, "p": 11.0, "c": 1.4},
    ...
  ],
  "central": {"h0": 1.0, "t": 3.1, "c0": 1.0},
  "budget": 812.5,
  "tau": 0.90,
  "lam": 0.1,
  "n_scenarios": 200
}
```

- `sites[i].mean`, `.std`, `.dist` — the demand (specimen-count) distribution at site
  `i`. `dist` is one of `"normal"`, `"lognormal"`, `"bimodal"`; demand is clamped at 0.
- `sites[i].h` — holding cost per **unused** kit left at site `i`.
- `sites[i].p` — shortage penalty per specimen **missed** at site `i` (typically `p >> h`).
- `sites[i].c` — acquisition cost per kit **placed** at site `i`.
- `central.h0`, `central.t`, `central.c0` — vessel holding, transfer, and acquisition costs.
- `budget` — cap on total acquisition spend.
- `tau` — per-site target service level (no-stockout probability).
- `lam` — weight of the service-level penalty term.
- `n_scenarios` — number of hidden survey-day scenarios used to grade you.

### Answer (stdout)

```json
{"q": [q_0, q_1, ..., q_{N-1}], "q0": q0}
```

- `q_i >= 0` — kits pre-positioned at site `i` (fractional allowed).
- `q0 >= 0` — kits held on the vessel.
- All values must be finite and non-negative.

## Feasibility

The acquisition spend must not exceed the budget (relative tolerance `1e-6`):

```
sum_i c_i * q_i  +  c0 * q0   <=   budget
```

A missing/extra entry, a non-finite or negative value, a **budget violation**, a
crash, a timeout, or non-JSON output scores **0.0** on that instance. The budget uses
only public data, so you can check it yourself.

## Objective (minimize)

The judge draws `n_scenarios` **hidden** survey-day demand vectors `d` from the
declared distributions (the seed lives only in the judge). For each scenario `k`:

- local leftover at site `i`: `leftover_i = max(0, q_i - d_ik)` → holding `h_i * leftover_i`
- local shortfall at site `i`: `short_i = max(0, d_ik - q_i)`
- The vessel reserve `q0` is ferried to cover shortfalls **deterministically**, in
  decreasing order of `(p_i - t)`; only sites with `p_i > t` are helped, each covered
  kit costs `t` and removes one unit of shortage, until the reserve is exhausted.
- Uncovered shortfall costs `p_i` per unit; the reserve left unused costs `h0` per unit.

```
scenario_cost = sum_i [ h_i*leftover_i + p_i*unmet_i + t*covered_i ] + h0*unused_reserve
operating_cost = mean over scenarios of scenario_cost
```

A **service penalty** is added. Let `phat_i` = fraction of scenarios in which site
`i`'s **local** kits alone sufficed (`d_ik <= q_i`):

```
service_penalty = lam * sum_i max(0, tau - phat_i) * p_i * mean_i
obj = operating_cost + service_penalty          (MINIMIZE)
```

## Scoring

The judge computes an internal **baseline** objective `b` from the order-to-mean
policy (`q_i = mean_i`, `q0 = 0`). Your per-instance score is

```
r = min(1.0, 0.1 * b / max(obj, 1e-12))
```

so matching the baseline scores `~0.1`, and you must be roughly `10x` cheaper than the
baseline to approach `1.0` (the pooled critical-fractile optimum is far from that, so
even strong policies keep headroom below `1.0`). The reported **Ratio** is the mean of
`r` over all `12` instances (which include larger, tighter held-out reefs).

## Hints / strategies

- Order-to-mean is a weak baseline (~50% service); the single-echelon **critical
  fractile** `q_i = mean_i + Phi^{-1}(p_i/(p_i+h_i)) * std_i` already helps a lot.
- Respect the **service target** `tau` and exploit **risk pooling**: a shared vessel
  reserve covers whichever site happens to run short, so you can trim local stock.
- A robust approach is **sample-average approximation**: draw your own scenarios from
  the declared distributions and locally optimize `q`, `q0` against the exact
  composite objective, then rely on generalization to the held-out survey days.
