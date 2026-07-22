# Anticipatory Tolls: Steering the Commuter Equilibrium

A city has **E parallel routes** (edges) from one origin to one destination.
**N commuters** travel every round. Each edge `e` has a **convex latency
function** `l_e(x) = a_e*x^2 + b_e*x + c_e` (all coefficients `>= 0`): the more
flow `x` an edge carries, the slower it gets, and it gets slower faster and
faster as load grows (congestion).

You are the tolling authority. You do **not** get to react turn by turn — you
must commit, in one shot, to a full **T-round toll schedule** before the
episode plays out. Every round, the commuters re-split themselves among the
edges via a fixed, deterministic **selfish best-response rule**: each edge's
*perceived cost* this round is last round's realized latency plus whatever
toll you posted for it now; more commuters flow toward whichever edge looks
cheap, in direct proportion to how cheap it looks. Tolls are a steering price
only — they are **not** counted as real travel time.

**The trap.** A controller reacting to whichever edge was jammed *last* round
(taxing today's hotspot) is always one step behind: taxing today's jam just
shoves the crowd onto a parallel edge, which becomes tomorrow's jam. The
winning insight is to work out the **system-optimal** split up front — the
split minimizing total travel time, found by equalizing every used edge's
*marginal* cost — and price the crowd's own best-response into landing there
directly, instead of chasing the hotspot.

## Dynamics (exact; you can simulate this yourself)

Given your toll schedule `tolls[t][e]` for rounds `t = 0..T-1`:

```
x^0 = x0                                            (given)
for t in 0..T-1:
    cost_e   = l_e(x_e^t) + tolls[t][e]
    share_e  = (1/cost_e) / sum_j(1/cost_j)
    x_e^{t+1} = (1 - rho) * x_e^t + rho * N * share_e
    realized_t = sum_e  x_e^{t+1} * l_e(x_e^{t+1})    (true travel time this round)
objective = sum_t realized_t                          (MINIMIZE)
```

`rho` (given) is the fraction of the population that reconsiders its route
each round.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. Runs in an isolated
subprocess; sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a toll schedule ...
print(json.dumps({"tolls": tolls}))
```

### Public instance (stdin)

```json
{
  "name": "b1", "E": 2, "N": 700.0, "T": 18, "rho": 0.25,
  "edges": [{"a": 0.004, "b": 0.05, "c": 1.0}, {"a": 1.0, "b": 0.05, "c": 1.0}],
  "x0": [700.0, 0.0],
  "toll_max": 123456.0
}
```

### Answer (stdout)

```json
{ "tolls": [[0.0, 12.3], [4.1, 9.0], ...] }   // T rows, each of length E
```

- `tolls` must be a list of exactly **T** rows, each a list of exactly **E**
  finite numbers with `0 <= tolls[t][e] <= toll_max`.
- Any malformed output (wrong shape, non-finite, out of range), a crash, a
  timeout, or non-JSON output makes that instance score `0.0`.

## Objective

**Minimize** total realized travel time, summed over rounds, over a fixed
seeded family of **10 instances** (varying `E`, `N`, `T`, `rho`, and the
edges' congestion coefficients). Some instances use a high `rho`, where an
edge that looks cheap this round pulls a large share of the crowd next round —
reactive tolling tends to just relocate the jam. Two instances are larger,
held-out cases for generalization.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `B` = total realized travel time under the **all-zero** toll schedule,
- `obj` = total realized travel time under **your** toll schedule,

and normalizes:

```
r = clamp( 0.2 * B / obj, 0, 1 )
```

Doing nothing scores `~0.2`; cutting total travel time below the no-toll
baseline scores above `0.2`. The **Ratio** reported is the mean of `r` over
all instances; the **Vector** lists the per-instance scores.

## Suggested strategies

1. **Do nothing** (baseline): zero toll every round.
2. **Reactive congestion pricing**: tax whichever edge was priciest last
   round, proportional to the excess over the cheapest edge right now.
3. **Anticipatory marginal-cost pricing**: solve for the system-optimal split
   (equalize marginal cost via water-filling), then post the constant toll
   that makes it a fixed point of the crowd's best-response.
4. **Local search**: perturb a constructive schedule per-round, per-edge to
   shave further time off the total.
