# Right Basin First: Regime-Aware Tour Building

You are given a set of `n` points in the plane. They came from one of three
kinds of layout — **strongly clustered** (tight groups scattered far apart),
**roughly uniform**, or a **perturbed grid** — but the instance does **not**
tell you which; you only get the raw coordinates. Build a single closed tour
visiting every point exactly once, minimizing total Euclidean length.

You do not have to finish the job yourself: after you submit an initial
order, the evaluator polishes it with its own bounded local search. But that
polish is capped, so the basin your order starts in matters more than how it
gets finished.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your visiting order) to **stdout**.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide a visiting order ...
print(json.dumps({"tour": [3, 0, 7, 1, ...]}))   # a permutation of 0..n-1
```

### Public instance (stdin)

```json
{
  "name": "clusA",
  "n": 80,
  "points": [[12.4, 55.1], [13.0, 54.7], ...],   // n [x, y] pairs
  "refine_budget": 6                              // see "Scoring" below
}
```

### Answer (stdout)

```json
{ "tour": [0, 5, 2, 1, 3, ...] }
```

`tour` must be a permutation of `0..n-1` (a closed loop; the last point
connects back to the first). Wrong length, a repeated or out-of-range index,
a non-integer, a crash, a timeout, or non-JSON output scores that instance
`0.0`.

## Scoring (deterministic)

The evaluator does **not** just measure your tour as submitted. It applies
its own fixed local search to it first:

```
final_tour = refine(your_tour, budget = refine_budget)
final_len  = length(final_tour)
```

`refine` performs up to `refine_budget` *accepted* improving moves, drawn
from a standard **2-opt** (reverse the segment between two edges) and
**Or-opt** (relocate a run of 1–3 consecutive stops elsewhere) neighborhood,
scanned in fixed order, until the budget is spent or no improving move
remains. The budget is intentionally small — enough to smooth local
wrinkles, nowhere near enough to redesign a bad global plan from scratch.

The evaluator also computes two internal reference lengths for normalizing
the score (never exposed to your program): `q_base`, from refining the plain
input-order tour with the same budget, and `q_target`, a stronger anchor —
the best length reachable from several constructions refined with a much
larger budget. Your score on that instance is

```
r = clamp( 0.1 + 0.75 * (q_base - final_len) / max(q_base - q_target, eps), 0, 1 )
```

Matching `q_base` scores `~0.1`; reaching `q_target` caps around `0.85`
(room stays open above any reference solution). **Ratio** is the mean of `r`
over 10 seeded instances; **Vector** lists the per-instance scores. Several
instances are strongly clustered on purpose.

## Why the obvious approach struggles

The textbook move is nearest-neighbor construction: start anywhere, always
walk to the closest unvisited point. It needs no knowledge of the layout —
which is exactly its weakness. On a strongly clustered instance,
nearest-neighbor greedily empties out whatever cluster it's in, then must
eventually jump clear across the map to an early-skipped, far-away cluster.
Each such jump is one bad edge; with enough clusters, the bad-edge count can
exceed the local search's small move budget, so several long jumps never
get fixed.

A layout-aware approach can do better by first asking a cheap question: *is
this point set clustered, uniform, or grid-like?* A quick signal is the
**nearest-neighbor-distance histogram** — compare each point's distance to
its closest neighbor against what you'd expect for a uniform random layout
of the same density. Clustered layouts pack points far closer together than
that; very regular grids sit farther apart with almost no spread. Once you
know the regime, pick a constructor built for it:

- **Clustered**: cluster first (e.g. cut the long edges out of a minimum
  spanning tree), solve each small cluster cheaply, then chain the clusters
  together — choosing *where* you enter and leave each cluster so the
  connecting edges stay short, rather than stitching independently-built
  sub-tours together at an arbitrary point.
- **Uniform / grid**: a space-filling sweep (e.g. sort into vertical strips,
  alternating scan direction) avoids the occasional long backtrack that
  nearest-neighbor can still produce even without clustering.

Since `refine_budget` is given to you, you can replay the evaluator's own
bounded refine locally on more than one candidate construction and keep
whichever comes out ahead — the budget rewards starting in the right basin,
not any particular constructor by name.
