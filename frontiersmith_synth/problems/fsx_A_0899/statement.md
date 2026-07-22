# Cable Trucks to the Substations

You are wiring `N` substations into a power network. A cheap **backbone** of
cables is already installed for free — two well-knit clusters joined by a
single thin bridge, so the network starts connected but has an obvious
weak point. Extra candidate cables arrive **one truck at a time**, `M`
trucks in total, each truck offering exactly one cable at a fixed cost.
You have a total cash **budget**.

This is genuinely online: your program is run **once per truck**, in
arrival order. On each run you are shown only *this* truck's cable
`(u, v, cost)`, the network you have built so far (backbone plus every
cable bought at earlier trucks), how much budget remains, how many trucks
there are in total (`M`) and which one this is (`t`), and one thing that is
entirely yours: a private JSON `state` blob you handed yourself on the
previous truck (the judge only stores and replays it back to you verbatim —
it never looks inside it). You answer accept/reject for *this* truck only,
immediately and permanently. There is no way to see the next truck before
deciding on this one, and no way to revisit a past decision — "I'll take
truck #40's cable" is not a plan you can make in advance; you only get it
if you actually still have the budget when truck #40 shows up.

Your goal: maximize the final network's **algebraic connectivity**
(`lambda_2`, the second-smallest eigenvalue of the graph Laplacian `L = D -
A`) — a standard measure of how well the network survives a single link
failure. `lambda_2` never decreases as edges are added, so buying a cable
never directly hurts; the only real cost is spending budget on a mediocre
cable that a better one later could have used.

## Candidate program contract

Standalone program, run **once per truck**: read ONE JSON object from
**stdin**, write ONE JSON object to **stdout**. Runs isolated each time.

```python
import sys, json
step = json.load(sys.stdin)
# ... decide, using step["state"] as your own memory of earlier trucks ...
print(json.dumps({"accept": True_or_False, "state": my_new_state}))
```

### Public per-step instance (stdin)

```json
{
  "n": 16, "m": 30, "t": 12,          // N substations, M trucks total, this is truck t
  "budget_total": 10.0, "remaining": 6.0,
  "backbone": [[0,1],[1,2], ...],      // free edges, same every call
  "accepted": [[3,9], ...],            // cables you bought at steps 0..t-1
  "u": 5, "v": 13, "cost": 3,          // THIS truck's cable
  "state": {"...": "..."}              // whatever you returned last call, or null at t=0
}
```

### Answer (stdout)

```json
{ "accept": true, "state": {"...": "..."} }
```

`accept` must be `true`/`false` (or `0`/`1`); `state` may be any JSON value
up to 4000 characters serialized — your private scratchpad, carried forward
call to call. Any other shape, a crash, a timeout, or non-JSON output at
**any** truck invalidates the whole instance, scoring it `0.0`.

## Scoring (deterministic)

For each instance the evaluator computes, itself, from the full manifest:

- `lam_base` — the `lambda_2` reached by a **myopic-greedy** reference: walk
  the manifest causally and buy truck `t` iff it is affordable right now
  *and* strictly increases the network's exact `lambda_2` built so far.
- `lam_ceil` — the `lambda_2` reached by buying **every** truck's cable,
  ignoring the budget entirely. Since `lambda_2` is monotone non-decreasing
  under edge addition, this is an unreachable ideal.
- `lam_cand` — the `lambda_2` of the network your accept/reject stream built.

```
r = clamp( 0.1 + 0.9 * (lam_cand - lam_base) / max(1e-9, lam_ceil - lam_base), 0, 1 )
```

Matching myopic-greedy scores ≈ `0.1`; reaching the (unreachable) ceiling
scores `1.0`; doing worse than myopic-greedy scores below `0.1`. No
budget-respecting policy saturates the ceiling — there is real headroom
above a good solution. The reported **Ratio** is the mean of `r` over 10
fixed instances; the **Vector** lists the per-instance scores.

## Why "accept anything that helps" is a trap

Most early trucks offer cheap cables **inside** one of the two clusters:
each gives a real, positive, but small `lambda_2` gain (it hardens a
cluster that is already fairly well connected). A handful of trucks
concentrated in the **last quarter** of the manifest offer cables that
**cross** the thin bridge — these attack the actual bottleneck and are
worth far more. Buying every early cable that helps is individually
correct at the moment you decide, yet collectively burns the budget before
the high-value late trucks arrive, and by the time you would want to
change your mind it is already too late — you already answered.

## Suggested strategies

1. **Reject everything** (baseline): keep only the free backbone.
2. **Myopic-greedy**: buy whatever is affordable and strictly helps, right now.
3. **Fixed absolute threshold**: only buy if the immediate gain clears some
   hand-picked bar — still ignores how much budget or manifest is left.
4. **Spectral leverage + adaptive pacing**: get a cheap Fiedler-vector
   sketch of the network built so far, score this truck's cable by
   `(x_u - x_v)^2 / cost`, spend an initial slice of trucks purely
   observing that distribution (buying nothing), then track — using your
   own `state` and the remaining-budget/remaining-truck-count you are
   handed each call — an acceptance percentile that relaxes as the
   manifest runs low, so money is not left unspent when the last truck
   pulls away.
