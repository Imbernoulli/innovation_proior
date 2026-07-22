# Firebreaks Against a Lattice Arsonist

An arsonist will strike **one** ignition point on a forest, then walk away. You do
not know which. Before they act you may pre-cut a limited number of **firebreak**
cells. Cut wisely so that, no matter where they light the match, as much fuel as
possible survives.

## Setup

The forest is an `N x N` grid. Cell `(r, c)` holds `fuel[r][c] >= 1`. Let `T` be the
total fuel. The arsonist's candidate ignition points are a fixed `K = 64` **lattice**
(an `8 x 8` grid of points spread across the map) — every one is a live grid cell,
given in the input.

You output a set of firebreak cells. A firebreak cell is cut: it holds no fuel and
fire cannot pass through it. Fire lit at ignition point `p` spreads to 4-neighbouring
cells that still have fuel and are not firebreaks; it burns the entire connected
component of un-cut cells containing `p`. Let `burned(p)` be the total fuel in that
component.

The arsonist maximises damage, so they pick the **worst** point:
`worst = max over the K ignition points of burned(p)`.
Your **protected fuel** is `F_obj = T - worst`. You want to make `F_obj` large.

## Input (stdin)

```
N F K
r1 c1            (K lines: the ignition-lattice points, 0-indexed)
...
rK cK
row 0            (N lines, each N integers: the fuel grid)
...
row N-1
```
`24 <= N <= 42`, `F` is your firebreak budget, `K = 64`.

## Output (stdout)

```
M
r1 c1            (M lines: your firebreak cells, 0-indexed, distinct)
...
rM cM
```

## Feasibility

`0 <= M <= F`; every cell in-grid and distinct; **no firebreak may sit on an
ignition point**. Any violation scores `0`.

## Objective & Scoring

Maximise the protected fuel `F_obj = T - worst`. The checker also builds a baseline
`B` = the protected fuel of a single straight wall cutting the map down the middle
(a naive halving). Your score is

```
Ratio = min(1.0, 0.1 * F_obj / B)
```

so the middle-wall baseline scores `~0.1` and you must partition far better to climb.

## What makes it hard

Because the ignitions form a **dense lattice**, essentially every compartment you
carve contains one — the arsonist can always reach your biggest compartment. So the
worst case equals the **most-fuelled compartment**, and the task is a *balanced
partition* problem: spend firebreak walls to equalise fuel across compartments, not
to ring hot spots or slice equal AREAS. The fuel is planted in small, dense blocks,
so a uniform equal-area cut leaves one compartment carrying most of the fuel, while
a cut placed at the fuel's own quantiles splits the dense mass apart.

## Example scoring

If `T = 8000` and your firebreaks leave a worst compartment of `2000` fuel, then
`F_obj = 6000`; if the middle-wall baseline leaves `worst = 5000` (`B = 3000`), your
`Ratio = min(1, 0.1 * 6000 / 3000) = 0.2`.

## Constraints

Time limit 5s, memory 512MB. Scoring is deterministic.
