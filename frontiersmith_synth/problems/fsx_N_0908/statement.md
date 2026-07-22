# Deterrence Beat: Staffing the Review Queue Against an Unpredictable Swarm

A trust & safety team reviews abuse across `K` categories (fake accounts, payment fraud,
phishing, spam, account takeover, review manipulation, ...). Each of `T` rounds, the queue
gets a limited review-throughput budget `R[t]` split across categories. The attacker
population is adaptive: strong *effective* coverage this round partly deters a category's
mass, which **migrates** per a fixed incentive graph `M` (`M[i][j]` large = cheap switch).
On top of this predictable drift, **every round an unpredictable fraction of each category's
mass takes one more opportunistic hop along the same graph `M`** — a shift you cannot forecast
from the deterministic law alone (its size/timing are hidden; only the mechanism, via `M`, is
public).

Submit the **entire T-round plan in one shot** (no live back-and-forth): read the instance,
plan against the law, print the full allocation matrix. A plan that spends its whole budget
matching only the noise-free forecast is exposed wherever the hidden hops land.

## Candidate program contract

Standalone program: read ONE JSON object (stdin), write ONE JSON object (stdout).

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a plan ...
print(json.dumps({"alloc": alloc}))
```

### Public instance (stdin)

```json
{
  "name": "queue-cascade", "K": 6, "T": 24,
  "beta": 0.35,        // migration rate: fraction of a category's mass that migrates away
                        // per round, scaled by that round's EFFECTIVE coverage there
  "lam_waste": 0.6,     // penalty per unit of capacity poured far beyond need (see below)
  "cap_mult": 3.0,
  "value": [2.1, 4.4, ...],     // K harm weights per unit of caught fraud
  "p0": [0.31, 0.05, ...],      // K floats, sums to 1: initial attacker distribution
  "M": [[0.0, 0.6, ...], ...],  // K x K migration-incentive matrix; row i sums to 1 over
                                 // j != i, M[i][i] = 0
  "V": [143.2, 148.9, ...],     // T floats: total attack volume that round (noise-free)
  "R": [78.8, 81.0, ...]        // T floats: review budget that round
}
```

### Answer (stdout)

```json
{"alloc": [[r_00, ..., r_0,K-1], ..., [r_{T-1},0, ..., r_{T-1},K-1]]}
```

`alloc` must have exactly `T` rows of exactly `K` non-negative finite numbers each, with each
row summing to at most `R[t]`. Any shape/type violation, a `nan`/`inf`, a negative entry, or a
round that overspends its budget makes the WHOLE submission score `0.0`.

## Dynamics and objective (evaluator replays this itself)

Starting from `p = p0`, each round `t`, for every category `j`: volume is `vol_j = V[t]*p[j]`;
you catch `min(alloc[t][j], vol_j)` units, worth `value[j]` each; coverage
`cov_j = min(1, alloc[t][j] / vol_j)`. A fraction `beta*cov_j` of `p[j]` migrates out,
redistributing into `k` weighted by `M[j][k]`; the rest stays. THEN an unpredictable extra
fraction of each category's (post-migration) mass takes one more `M`-weighted hop — its size
is seeded per round but hidden from you. Capacity spent on `j` beyond
`max(vol_j, cap_mult*R[t]/K)` is charged `lam_waste` per unit (over-staffing far past both
live volume and a reasonable proactive margin is wasteful).

**Maximize**, across all `T` rounds: `sum(prevented harm) - lam_waste * sum(waste)`.

## Scoring

The evaluator computes two internal references purely from the instance (never seen by you as
a target): a uniform, non-adaptive split (`q_base`), and an inflated near-ceiling `q_top` built
from two internal policies (one that spends its whole budget matching the noise-free forecast,
one that reserves a steady migration-graph-weighted floor), then normalizes:

```
r = clamp( 0.1 + 0.9 * (your_score - q_base) / (q_top - q_base), 0, 1 )
```

Matching the uniform split scores ≈ 0.1; doing worse scores below 0.1. `q_top` is inflated
above both internal references, so even a strong policy stays below 1.0 — there is real
headroom. Reported **Ratio** is the mean `r` over 10 seeded instances (some larger / held-out
for generalization); **Vector** lists the per-instance scores.

## Suggested strategies

1. **Uniform** (baseline): split every round's budget evenly, ignore everything else.
2. **Forecast-matching**: each round, allocate proportional to `volume x value` under the
   noise-free law, self-simulated forward from `p0`. Well-informed, but fully exposed to the
   unpredictable hop every round.
3. **Structural floor**: power-iterate `M` (weighted by `value`) to find where migration
   pressure structurally accumulates, and reserve a fixed share of the budget there — every
   round, regardless of that round's forecast.
4. **Structural floor + stability**: as above, but blend the forecast top-up slowly toward
   last round's plan so coverage never lurches to chase a single round's forecast wiggle.
