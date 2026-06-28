# Simulated Epidemic Containment (control loop)

## Research question

There are `n` regions connected by a weighted contact graph. A disease spreads through them by a
**deterministic SIR-on-a-graph** model: each region `r` carries three compartment fractions
`(S_r, I_r, R_r)` with `S_r + I_r + R_r = 1` (susceptible / infected / recovered), and the epidemic
evolves over `T` discrete days. Each day, **before** the spread step, a controller may **lock down**
up to `b` regions; a locked region's transmission — internal and across every incident edge — is
multiplied by a residual factor `kappa in [0,1)` for that one day. The task is to choose the per-day
lockdown set so as to **minimize the total number of new infections** summed over all regions and all
`T` days.

This is a finite-horizon optimal-control problem over a nonlinear epidemic. The state space is
continuous and high-dimensional (`3n` compartment values), the per-day action space is "any size-`<=
b` subset of `n` regions", and the dynamics are deterministic but nonlinear, so there is no exact
closed-form optimum to read off; the quality of a schedule is judged by a continuous score. The only
lever is the heuristic that decides, day by day, which regions to lock. Because `b` is small relative
to `n` (you can only cover a handful of regions per day), the budget is binding and the choice
matters.

## Input / output contract

- Input (stdin):
  - first line: `n m T b` — number of regions `n` (`30 <= n <= 60`), number of contact edges `m`,
    horizon length `T` (`16 <= T <= 24`) in days, and the **daily** lockdown budget `b`
    (`4 <= b <= 6`);
  - second line: `beta gamma kappa` — three reals: the transmission rate `beta`, recovery rate
    `gamma`, and the residual-transmission factor `kappa` of a locked region (all read as doubles);
  - then `m` lines, each `u v w` — an **undirected** weighted contact edge between regions `u` and
    `v` (`0 <= u, v < n`, `u != v`) with weight `w in (0, 1]`. The graph is connected and simple;
  - then a final line of `n` reals `I0_0 .. I0_{n-1}` — the **initial infected fraction** of each
    region (`I0_r in [0,1]`, most are `0`).
- Output (stdout): a **lockdown schedule** of exactly `T` lines. Line `t` (day `t`, 0-indexed) starts
  with an integer `c_t` (how many regions are locked that day) followed by `c_t` **distinct** region
  ids in `[0, n)`. Tokens are whitespace-separated; line breaks do not matter for parsing — the
  grader reads, for each of the `T` days in order, one count `c_t` then `c_t` ids, and the total
  token count must be exactly `sum_t (1 + c_t)` with no leftover.
- Time limit: about 2 seconds. Memory: 256 MB.

Example shape: with `T = 3, b = 2`, a valid output is three lines such as `2 4 7` / `0` / `1 4` —
day 0 locks regions 4 and 7, day 1 locks nobody, day 2 locks region 4. Each line has at most `b`
distinct ids.

## Background

The state evolution is the standard discrete-time, network SIR with a force-of-infection that mixes
a region's own infected fraction with that of its graph neighbours. Writing `factor_r = kappa` when
region `r` is locked today and `factor_r = 1` otherwise, the **force of infection** on region `r` on
a given day is

```
lambda_r = factor_r * ( beta * I_r  +  sum over neighbours (j, w) of r  of  w * beta * I_j * factor_j )
```

— so locking either endpoint of an edge damps the transmission across it. The number of **new
infections** in `r` that day is `newinf_r = S_r * (1 - exp(-lambda_r))` (the exponential keeps this in
`[0, S_r]`, so the susceptible pool never goes negative), and `newrec_r = gamma * I_r` people recover.
All regions are updated **simultaneously** from the same pre-update `I` values:
`S_r -= newinf_r; I_r += newinf_r - newrec_r; R_r += newrec_r`. The objective accumulates
`sum_r newinf_r` over all `T` days.

Two reference controllers frame the design.

- **Lock the most-infected regions today (myopic greedy).** Each day, lock the `b` regions with the
  largest current infected fraction `I_r`. This is the obvious reflex and it is genuinely reasonable —
  it directly suppresses the regions that are radiating the most infection *right now*. It is the
  baseline the scorer normalizes against. Its weakness is that it is **myopic**: a region that is
  hot today may already be burning out (little `S` left to infect), so locking it averts little,
  while a *still-susceptible* region sitting just ahead of the wavefront — not yet the most infected —
  is exactly where a lockdown would prevent a large future outbreak. The myopic rule never sees that.
- **Rolling-horizon look-ahead control.** Instead of ranking regions by their current `I`, score a
  candidate lockdown by how many infections it would **avert over a short future horizon**: roll the
  current epidemic state forward `H` days under a cheap default future policy, once with the candidate
  left open and once with it locked today, and take the drop in projected cumulative new infections as
  the candidate's value. This is the established model-predictive-control idea for epidemic
  intervention — use the model you already have to simulate a few steps ahead and let the *projected*
  marginal effect, not the instantaneous state, drive the decision. The open design choices are the
  horizon length, the default future policy used inside the rollout, and how to choose `b` interacting
  regions per day (a flat top-`b` by individual look-ahead value double-counts overlapping benefit).

## Evaluation settings

A solution is first checked for **feasibility**; any violation floors the score to **0**:

1. the schedule parses as the `T` blocks `(c_t, id_1, ..., id_{c_t})` above, with **no missing and no
   leftover tokens** (the total token count is exactly `sum_t (1 + c_t)`);
2. every `c_t` is an integer with `0 <= c_t <= b` (the **daily** budget — exceeding it on any single
   day is illegal);
3. every region id is an integer in `[0, n)`;
4. within a single day the locked ids are **distinct** (no region locked twice on the same day).

For a feasible schedule the scorer runs the **full deterministic SIR simulation** above, applying the
schedule's locks day by day, and computes `total_new_infections = sum over days and regions of
newinf_r` (lower is better). The score normalizes against a deterministic **"lock the `b`
most-infected regions today"** greedy baseline that the scorer recomputes itself:

```
score = round(1_000_000 * baseline_infections / max(1e-9, solver_infections))     (0 if INFEASIBLE)
```

So the myopic baseline scores about `1_000_000`; a schedule that lets through **fewer** infections
scores more, and an infeasible one scores `0`.

**Instances** are generated deterministically from an integer seed. `n` is in `[30, 60]`; the contact
graph is a random spanning tree (guaranteeing connectivity) plus extra random edges, average degree
about `3`–`5`, edge weights in `(0, 1]`. `T` is in `[16, 24]` and `b` in `[4, 6]`. The rates are
chosen so the basic reproduction number sits roughly in `1.5`–`2.5`: an uncontrolled epidemic still
grows, but the budget is large enough that smart containment can genuinely **suppress** it rather than
merely **delay** it. (If `R0` were far above `1` with a long horizon the epidemic would saturate no
matter what the schedule is, and the control choice would not matter — that regime is deliberately
avoided.) A handful of seed regions start with a small infected fraction and the rest start almost
fully susceptible, so the epidemic must travel across the graph — which is exactly what makes a
look-ahead beat the myopic "lock the most-infected-today" rule.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes a feasible
schedule to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, T, b;
    if (scanf("%d %d %d %d", &n, &m, &T, &b) != 4) return 0;
    double beta, gamma, kappa;
    scanf("%lf %lf %lf", &beta, &gamma, &kappa);

    vector<vector<pair<int,double>>> g(n);
    for (int e = 0; e < m; ++e) {
        int u, v; double w; scanf("%d %d %lf", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
    }
    vector<double> I0(n);
    for (int r = 0; r < n; ++r) scanf("%lf", &I0[r]);

    // A feasible fallback: lock nobody on any day (every c_t = 0). This always
    // parses and never exceeds the budget, so it scores > 0 -- a valid safety net.
    // TODO heuristic: a ROLLING-HORIZON controller. Each day, score a candidate
    // lockdown by the new infections it AVERTS over a short look-ahead horizon
    // (roll the cached SIR state forward H days, with vs. without the candidate
    // locked today), and build the day's <= b locks by GREEDY MARGINAL GAIN so
    // the interacting picks do not double-count overlapping benefit.
    vector<vector<int>> schedule(T);  // schedule[t] = ids locked on day t

    for (int t = 0; t < T; ++t) {
        int c = (int)schedule[t].size();
        printf("%d", c);
        for (int id : schedule[t]) printf(" %d", id);
        printf("\n");
    }
    return 0;
}
```
