# One Release Calendar for Wet and Dry Years

## Problem

A dam operator must publish **one** weekly release calendar `r_1, ..., r_T` (tonnes of
water per week) that will be used for `T` weeks, **before** knowing which of `K` possible
inflow trajectories ("scenarios" — think historical wet/dry years) actually happens. The
calendar is fixed; only the weather scenario varies.

Reservoir storage starts at `S0`. Under scenario `k`, storage evolves as

```
S_0^k = S0,     S_t^k = S_{t-1}^k + inflow_t^k - r_t     (t = 1..T)
```

**Feasibility (hard, per scenario):** for *every* scenario `k` and *every* week `t`, storage
must stay within `[D, C]` (dead pool to full capacity) and every weekly release must satisfy
`0 <= r_t <= Rmax`. Violating this in even **one** scenario at **one** week makes the whole
calendar infeasible, regardless of how it performs elsewhere.

**Power.** In week `t`, scenario `k` generates

```
power_t^k = shape(r_t) * head(S_{t-1}^k)
shape(r)   = r - r^2 / (2.2 * Rmax)                        (concave in the release itself)
head(S)    = HMIN/1000 + (HMAX-HMIN)/1000 * (2x - x^2),  x = clip((S-D)/(C-D), 0, 1)
                                                            (concave, increasing in storage)
```

Both curves have diminishing returns: bursty single-week dumps and low-storage operation
both cost you power, even for the same *total* volume released.

**Objective.** Maximize the **worst scenario's** total power:
`F = min_k sum_{t=1..T} power_t^k`. A calendar that looks great on average but ruins the
wettest or driest year scores by its worst year, not its average one.

There is no known closed-form optimum: it depends on the exact shape of every scenario's
inflow trajectory over time, not just totals or averages.

## Input (stdin)

```
T K C D S0 Rmax HMIN HMAX
inflow^0_1 ... inflow^0_T
...
inflow^{K-1}_1 ... inflow^{K-1}_T
```
All values are non-negative integers; `HMIN`, `HMAX` are per-mille (divide by 1000 for the
head multiplier bounds).

## Output (stdout)

Exactly `T` whitespace-separated integers `r_1 ... r_T` — the single calendar applied to
every scenario.

## Feasibility

Output invalid (score 0) if: the token count isn't exactly `T`; any token isn't a finite
integer; any `r_t` falls outside `[0, Rmax]`; or, simulating **any** scenario `k`, storage
`S_t^k` ever leaves `[D, C]`.

## Scoring

Let `F` be your worst-scenario total power (feasible outputs only). The checker builds its
own always-feasible reference calendar `B` — a "wait, then release the legal maximum in one
burst" schedule that satisfies the same hard corridor but ignores the concavity of `shape`
and `head`. Then

```
Ratio = min(1000, 100 * F / max(1e-9, B)) / 1000
```

so `Ratio` lies in `[0, 1]`; matching `B` gives `~0.1`, and beating it 10x saturates at `1.0`.

## Example (illustrative shape only — not a scored case)

With `T=3, K=2, C=100, D=20, S0=50, Rmax=40, HMIN=100, HMAX=1000`, scenario A inflow
`[10,10,10]`, scenario B inflow `[10,60,10]`: any calendar releasing `r=[0,30,0]` keeps A in
`[20,70]` and B in `[20,90]` — feasible for both — while `r=[0,0,0]` overflows B in week 2
(`50+10+60=120 > 100`). Among feasible calendars, the one that also keeps storage high in
week 1 (before B's spike is even visible) tends to score best, because it does not sacrifice
head early to protect against a threat that only some scenarios carry.

## Constraints

`4 <= T <= 60`, `2 <= K <= 12`, `1 <= D < C <= 10^6`, `D <= S0 <= C`, `1 <= Rmax <= 10^6`,
`0 <= inflow_t^k <= 10^5`, `100 <= HMIN < HMAX <= 1000`. Time limit 5s, memory 512MB.
