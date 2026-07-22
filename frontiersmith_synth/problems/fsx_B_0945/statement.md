# Rotational Grazing on a Paddock Grid

A herd grazes an `R x C` grid of paddocks (indexed `0..R*C-1`, row-major). Every
paddock holds a standing-grass level `g` in `[0, 1]`. Each day, **whether or
not the herd visits**, every paddock's grass evolves by logistic regrowth plus
diffusion with its up-to-4 grid neighbours:

```
g[i] += r * g[i] * (1 - g[i]) + D * sum_{j in neighbours(i)} (g[j] - g[i])
g[i]  = clip(g[i], 0, 1)
```

`r` is the regrowth rate; `D` is the diffusion rate. Diffusion cuts both ways:
a bare paddock next to lush neighbours gets reseeded quickly, but a paddock
that keeps "lending" grass to a bare neighbour is dragged down too.

Each day the herd occupies exactly one paddock and eats up to a fixed daily
requirement `D_req` from it: `eaten = min(g[c], D_req)`, applied to `g[c]`
*before* that day's regrowth/diffusion step. Any shortfall `D_req - eaten`
must be covered with supplemental feed, at cost `supp_mult` per unit.
Moving the herd to a different paddock than the day before costs
`move_fixed + move_per_dist * manhattan_distance` (staying costs nothing).

## Your job

Write a **standalone program**: read one JSON public instance from **stdin**,
write one JSON answer to **stdout**. It runs isolated and only sees the public
instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide the whole season's visit sequence ...
print(json.dumps({"visits": visits}))
```

### Public instance (stdin)

```json
{
  "name": "field_C", "R": 3, "C": 7, "T": 90,
  "r": 0.14, "D": 0.03, "D_req": 0.22,
  "move_fixed": 0.06, "move_per_dist": 0.04, "supp_mult": 1.6,
  "start": 0,
  "g0": [0.9, 0.14, ...]        // length R*C, initial grass per paddock
}
```

Everything the simulation needs is public -- you can replay the exact
dynamics yourself to plan ahead.

### Answer (stdout)

```json
{ "visits": [0, 0, 5, 5, 5, 12, ...] }   // length T; visits[t] in [0, R*C)
```

`visits[t]` is the paddock grazed on day `t` (day 0's "previous" location is
`start`). Wrong length, a non-integer or out-of-range index, a crash, a
timeout, or non-JSON output makes that instance score `0.0`.

## Objective

**Maximize**, over the season:

```
sum_t eaten_t  -  supp_mult * sum_t shortfall_t  -  sum_t move_cost_t
```

across a fixed, seeded family of 10 instances that vary in grid shape, season
length, regrowth/diffusion rates, relocation costs and initial grass layout
(uniform, patchy, one lush cluster, two far-apart lush clusters, or freshly
depleted). Two instances are larger, held-out cases.

## Scoring (deterministic)

For each instance the evaluator computes, itself, by replaying the *exact*
public formula above:

- `obj_base` = objective of the "never move" (stay at `start` forever) policy,
- `obj_ub` = `T * D_req` -- a loose, generally unreachable ceiling (full daily
  requirement, every day, forever, with zero relocation and zero supplement
  cost),
- `obj_cand` = objective of **your** visit sequence,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (obj_cand - obj_base) / max(eps, obj_ub - obj_base), 0, 1 )
```

Matching "stay put" scores ≈ `0.1`; reaching the unreachable ceiling scores
`1.0`; doing worse than staying put scores below `0.1`. Because `obj_ub`
ignores travel and diffusion limits entirely, even a strong schedule stays
well below `1.0` -- there is real headroom.

The reported **Ratio** is the mean of `r` over all 10 instances; the
**Vector** lists the per-instance scores.

## Suggested strategies

1. **Stay put** (baseline): never move; live off one paddock's own regrowth.
2. **Reactive greenest-first**: every day, move to whichever paddock currently
   has the most grass -- ignores relocation cost and the fact that grazing
   today reshapes tomorrow's diffusion neighbourhood. On fields with
   far-apart lush patches this thrashes the herd back and forth paying huge
   relocation fees; on fields with one diffusion-fed cluster it keeps
   re-draining the cluster (and its donor neighbours) until the whole
   neighbourhood collapses.
3. **Period-matched compact rotation**: estimate how long a grazed paddock
   needs to recover under the instance's own `r`/`D`, then tour a short,
   compact loop of paddocks sized so each is revisited on roughly that
   period.
4. **Local search**: refine loop membership, order and dwell time with a
   seeded multi-restart hill-climb over the replayed objective.
