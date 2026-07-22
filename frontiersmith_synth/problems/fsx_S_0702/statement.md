# Meridian County Micro-Depot Network

## Setting

Meridian County must open a batch of last-mile parcel **micro-depots** to serve
**N households/blocks**. Every household has a fixed `(x, y)` location and a daily
parcel **weight** (delivery volume). A depot can only be built at one of **M
candidate sites**, each with its own construction/lease **cost**. You must choose
exactly **K sites to open** and assign every household to one open depot.

The county's total daily cost is

```
cost = sum of opening costs of the K opened sites
     + sum over households of weight * (Euclidean distance to its assigned depot)
```

Minimize `cost`. There is no clean optimum: the county's settlement pattern has
**hidden multi-scale structure** — several coarse **districts**, each holding one
or more **tight population pockets** (dense apartment blocks / hamlets) at a finer
scale, plus households scattered thinly across open countryside. Candidate sites
exist near every pocket, near every district center, and scattered across the
countryside. Because `K` is smaller than "one depot per pocket", you must decide
*which* pockets deserve their own depot and which can share a coarser,
district-level depot — a genuine trade-off across scales, not a single clustering
pass.

## Input (stdin): ONE JSON object — the public instance

```json
{
  "name": "depot702001",
  "n_demand": 84,
  "n_sites": 23,
  "k": 5,
  "points": [[x, y, w], ...],   // length n_demand; household location + weight
  "sites":  [[x, y, cost], ...] // length n_sites; candidate site + opening cost
}
```

## Output (stdout): ONE JSON object

```json
{ "facilities": [s_0, ..., s_{K-1}], "assign": [f_0, ..., f_{N-1}] }
```

`facilities` must be exactly `K` **distinct** integers in `[0, n_sites)` — the
opened sites. `assign[i]` is the **position within `facilities`** (an integer in
`[0, K)`) of the depot serving household `i`.

A plan is **valid** iff both lists have the right length/type/range and
`facilities` has no duplicates. Wrong length, a duplicate or out-of-range site, a
crash, a timeout, or non-JSON output → that instance scores **0.0**. There is no
capacity limit on a depot — any assignment to an *opened* site is feasible, but
you pay its real distance.

## Scoring

Let `cost(plan)` be the formula above. Per instance the evaluator computes two
references entirely on its own (never seen by your program): `q_base`, the cost
of opening the `K` cheapest sites with every household sent to its nearest open
depot (a weak, geometry-blind reference), and `q_ref`, the cost reached by an
internal high-effort adaptive search with more restarts/iterations than your
program gets inside its time limit. Your plan's cost `q_cand` is normalized:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_ref), 0, 1 )
```

Matching the cheapest-sites baseline → `r ≈ 0.1`. Reaching the internal
high-effort reference → `r = 1.0` (generally unreachable in your time budget).
Doing worse than the baseline → `r < 0.1`. The final score is the **mean of `r`**
over 10 instances (a mix of medium and larger held-out counties).

## Isolation

Your program runs in an isolated subprocess: read the public instance from
stdin, write your answer to stdout. You never see the references, the scoring
internals, or any other candidate's output.

## Suggested strategies

1. **Cheapest-K** — ignore geometry, open the `K` lowest-cost sites, assign
   nearest. Weak but valid.
2. **Farthest-point sampling** (k-center greedy) — repeatedly open the site that
   maximizes the minimum distance to already-opened sites. Spreads depots across
   districts but, by construction, never opens two depots close together — so it
   can finish covering every district before ever splitting a dense pocket off
   from its district depot, even when that pocket's demand justifies it.
3. **Density-aware adaptive search** — diagnose each household's local demand
   scale (e.g. distance to its k-th nearest neighbor: small inside a tight
   pocket, large in open countryside); use that scale to size how aggressively
   you tear down and rebuild the current depot layout near the worst-served
   household — small, surgical moves inside dense pockets, larger sweeping moves
   in sparse areas — and accept moves under a shrinking cost-slack threshold
   (not just strict improvement) so you can escape the coarse, district-level
   layout that a single spread-out construction locks onto.
