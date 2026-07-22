# Protoplasm Wiring Diagram — Slime Mold Steiner Maze

## Problem

A slime mold (*Physarum polycephalum*) must wire together `K` city terminals
placed in an `R x C` grid maze that contains impassable wall cells. Instead of
directly submitting a network, you submit a **starting condition** for the
mold's own growth rule, and a deterministic simulation grows the final tubes
for you.

Every pair of orthogonally-adjacent non-wall cells is a candidate **tube
segment** with a conductance (thickness) `D >= 0`. You submit:

- a **feedback exponent** `mu` (how strongly a segment's conductance chases
  the flow it carries), and
- a sparse set of **initial thickness overrides** on specific segments
  (everything you don't mention starts at a fixed default `D_BASE = 0.30`).

The checker then runs a fixed number of reinforcement rounds (`ROUNDS = 8`).
In each round, for **every pair of terminals**, it solves the segment network
as an electrical circuit: inject 1 unit of current at one terminal, extract it
at the other (Kirchhoff's current law, conductances = current `D`), giving a
flux `Q_e` on every segment `e`. It **sums** `|Q_e|` over all terminal pairs
into `Qtot_e` (fluxes from different pairs add up on a shared segment before
anything else happens), then relaxes conductance toward that combined signal:
`D_e <- 0.6*D_e + 0.4*clip(Qtot_e, 0, inf)^mu`, clipped to `[0.02, 5.0]`.

After 8 rounds, the **tube network** is every segment with final `D_e >=
0.12`. This network must connect all `K` terminals (else you score 0). The
**objective is the number of unit-length segments in that network** — the
slime mold's total wiring length — which you want to **minimize**.

Why `mu` matters: at `mu = 1` the relaxation is linear and a uniformly seeded
maze decays almost uniformly, never concentrating flow onto a sparse
skeleton. At `mu > 1`, a segment carrying the *combined* flux of several
terminal pairs (a natural meeting point) gets a **super-additive** boost over
a segment used by only one pair — the only way a junction absent from every
pairwise shortest path can out-grow those paths.

## Input (stdin)

```
R C K
r_1 c_1 r_2 c_2 ... r_K c_K
W
or_1 oc_1 ... or_W oc_W
```
`R,C` grid size (rows, cols, 0-indexed). `K` terminals at distinct free cells.
`W` wall cells (obstacles); all other cells are free. The free-cell graph
(4-connectivity) is guaranteed connected and spans all terminals.

## Output (stdout)

```
mu
k
r1 c1 r2 c2 d0     (k lines)
```
`mu` in `[0.2, 4.0]`. `k` is the number of overrides (`0 <= k <= 4000`). Each
override line names an edge between two orthogonally-adjacent free cells and
its initial conductance `d0 in [0.02, 5.0]`. Unmentioned edges default to
`D_BASE = 0.30`. Later duplicate overrides of the same edge replace earlier
ones.

## Feasibility

Reject (score `0`) if: token counts don't match `k`; `mu`/`d0` is missing,
non-numeric, non-finite, or out of range; an override edge's endpoints are out
of bounds, not orthogonally adjacent, not both free cells, or not a real grid
edge; or the final simulated network fails to connect all `K` terminals.

## Objective & Scoring

Let `F` = number of segments in your final tube network (minimize). Let `B` =
the same quantity from the checker's own reference run (`mu = 1.0`, zero
overrides — the "do nothing" field). Then

```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```

Matching the baseline scores `0.1`; a network with a fifth of the baseline's
segments caps the score at `1.0`.

## Constraints

`7 <= R,C <= 13`, `2 <= K <= 6`, `0 <= W`. Time limit 5s, memory 512MB. The
checker is `O((R*C) * ROUNDS * K^2)` — one small dense linear solve per
terminal pair per round; fully deterministic given the same output.

## Example (worked score)

On a small 7x7 maze with 2 terminals and no walls, the checker's own
reference run (`mu=1.0`, no overrides) settles into a network of `B = 19`
segments — a lot of nearly-equal-conductance clutter since nothing was
seeded. A submission that reinforces the single shortest path between the two
terminals converges to `F = 6` segments (the direct route survives, the rest
decays below threshold). Score `= min(1000, 100*19/6) / 1000 = min(1000,
316.7)/1000 = 0.3167`.
