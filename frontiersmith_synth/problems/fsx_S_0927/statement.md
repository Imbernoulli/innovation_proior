# One Circuit, No Nemesis

A courier operates a single vehicle that must visit every stop in a territory
once and return to where it started (a closed circuit). You are given the
courier's toolbox of route-construction ideas, but you must pick or combine
them yourself: there is no time to run more than one construction per
territory, so the real decision is **which rule to trust for this layout** and
how to polish whatever it builds.

Territories are drawn from **five fixed layout families**: tight scattered
pockets, uniform open scatter, winding corridors that fold back on
themselves, radiating hub-and-spoke networks, and pairs of separated dense
grid blocks. **The family a given territory belongs to is never told to
you** — you only ever see raw stop coordinates. Any recipe that only works
well on the layouts it was written for will eventually meet its nemesis: a
family whose geometry breaks that recipe's assumptions.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your answer) to **stdout**.
It runs in an isolated subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a visiting order ...
print(json.dumps({"tour": order}))
```

### Public instance (stdin)

```json
{
  "name": "case03_hubspoke",
  "n": 35,
  "points": [[12.4, -3.1], [0.9, 44.2], ...]   // n stops, [x, y] floats
}
```

### Answer (stdout)

```json
{ "tour": [7, 2, 0, 34, ...] }   // a permutation of 0..n-1, length n
```

`tour[i]` is the index of the stop visited i-th; after the last stop the
vehicle returns to `tour[0]`, closing the loop. A tour is **valid** iff it is
a list of exactly `n` **distinct** integers, each in `[0, n-1]`. Any invalid
output (wrong length, a repeated or out-of-range index), a crash, a timeout,
or non-JSON output makes that instance score `0.0`.

## Objective

**Maximize** the reported score across a fixed, seeded set of **10**
instances (2 per family, 5 families). Crucially, the reported score is **not**
a plain average — it is the **minimum of the five per-family means**. A rule
that scores brilliantly on four families and collapses on the fifth is
scored exactly as if it always collapsed on that fifth family. One robust
dispatch rule beats a portfolio of specialists that never gets deployed
correctly.

## Scoring (deterministic)

For each instance the evaluator computes, itself, from the full point set:

- `L_lb`   = the **minimum spanning tree weight** of the stops — a real,
  generally-unreachable lower bound on any closed tour,
- `L_base` = the length of the tour that visits stops **sorted by
  x-coordinate only** — a weak, deterministic reference,
- `L_cand` = the length of **your** closed tour,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (L_base - L_cand) / max(1e-9, L_base - L_lb), 0, 1 )
```

Matching the x-sort reference scores ≈ `0.1`; reaching the MST lower bound
scores `1.0` (essentially unreachable); doing worse than the reference scores
below `0.1`. Because the MST bound is loose for most layouts, even excellent
tours stay well below `1.0` — there is real headroom.

Each of the 10 `r` values is grouped by its (hidden) generating family; the
**Ratio** you see is the minimum of the 5 family means, and **Vector** lists
all 10 per-instance ratios in generation order.

## Suggested strategies

1. **x-sort** (baseline): order stops by x-coordinate alone.
2. **Nearest-neighbor**: from a start stop, always hop to the closest
   unvisited stop. A solid single recipe — but it can get stranded: on a
   radiating network it walks one branch all the way to its tip, then must
   pay a huge jump back; on a folded corridor it can jump across the fold
   instead of following the path.
3. **Angular sweep**: order stops by polar angle around their centroid.
4. **Row-band sweep**: bucket stops into horizontal bands and traverse each
   band alternately left-to-right / right-to-left.
5. **Dispatch + polish**: compute cheap structural signals straight from the
   coordinates — e.g. how uneven each stop's nearest-neighbor distance is
   across the set, and what fraction of stops sit on the convex hull — to
   decide which construction above avoids this layout's failure mode, then
   run a shared local-search pass (e.g. 2-opt) on whichever tour you built.
