# Orbit Stitcher: Cheapest Multicolor Embroidery Run

An embroidery machine must stitch a multicolor design laid out on an integer
grid. The design is described as a set of required **unit stitch segments**,
each with a thread color. You control the needle with two kinds of operations
and must cover every segment exactly once at minimum total cost.

## Problem

The needle starts parked at vertex `0` with **no thread color loaded**.

- `S u v` — **stitch** the segment `{u, v}`. Legal only if the segment exists,
  has not been stitched yet, and the needle is currently at `u` or `v`.
  Costs `1`, plus `K` if the segment's color differs from the currently loaded
  color (loading the first color also costs `K`). The needle ends at the other
  endpoint.
- `J v` — **jump** the needle to vertex `v` without stitching. Costs the
  Manhattan distance `|x - x_v| + |y - y_v|` from the current needle vertex.

Your task: output an op sequence that stitches every segment exactly once,
minimizing the total cost `F`.

## Input (stdin)

```
V E C K
x_0 y_0
...
x_{V-1} y_{V-1}
u_1 v_1 c_1
...
u_E v_E c_E
```

- `1 <= V <= 3000`, `1 <= E <= 2500`, `1 <= C <= 8`, `1 <= K <= 30`.
- Coordinates are integers `0 <= x, y <= 10000`. Segments are unit grid edges;
  `{u_i, v_i}` pairs are distinct. Colors lie in `0..C-1`.

## Output (stdout)

```
M
op_1
...
op_M
```

`M` = number of ops (`0 <= M <= 200000`). Each op is `S u v` or `J v`.
No extra tokens.

## Feasibility

Every op must be legal (see above), every segment stitched exactly once, and
the op count must match the tokens exactly. Any violation, malformed token, or
non-integer value scores `0`.

## Objective and Scoring

Minimize `F` = total op cost. The checker builds its own baseline `B`: stitch
the segments **in input order**, jumping to `u` whenever the needle is at
neither endpoint. Then

```
Ratio = min(1, 0.1 * B / F)
```

so the input-order construction scores exactly `0.1`, and a plan `c` times
cheaper scores `0.1 * c`, capped at `1.0`.

## Example (illustrative; not an actual test)

Input:

```
4 4 2 10
0 0
1 0
1 1
0 1
0 1 0
3 2 0
1 2 1
0 3 1
```

A unit square: edges `{0,1}` and `{2,3}` have color 0, edges `{1,2}` and
`{0,3}` have color 1; `K = 10`.

Baseline `B` (input order): stitch `{0,1}` (load color 0: `10+1`), jump `1->3`
(`2`), stitch `{3,2}` (`1`), stitch `{1,2}` (color change: `10+1`), jump
`1->0` (`1`), stitch `{0,3}` (`1`). Total `B = 27`.

A better plan:

```
6
S 0 1
J 2
S 3 2
S 0 3
J 1
S 1 2
```

Trace: stitch color 0 first (`11`, needle at 1), jump `1->2` (`1`), stitch
`{2,3}` (`1`, needle at 3), switch to color 1 and stitch `{0,3}` (`11`, needle
at 0), jump `0->1` (`1`), stitch `{1,2}` (`1`). Total `F = 26`, so
`Ratio = min(1, 0.1 * 27 / 26) = 0.103846`.

## Constraints

- Time limit 2 s, memory 512 MB. Exact integer arithmetic; scoring is fully
  deterministic.
- The instances are assembled from **congruent copies of one multicolor motif
  placed far apart** (an orbit under a translation group), with the edge list
  scrambled. Each motif copy carries every color in fragmented bands. Locking
  to one color at a time minimizes color changes but sweeps the whole lattice
  once per color; sweeping region by region minimizes jump mileage but re-buys
  color changes. Where the balance lies is for you to discover — and the best
  plans also tune *how* each region is traversed (entry/exit chaining, trail
  decomposition), not just the region order.
