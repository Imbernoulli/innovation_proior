# The Spine, the Spurs, and the Vault

You explore a network of grid cells connected by one-way-usable corridors.
You start at a fixed cell and take a **self-avoiding walk**: at each step you
move to an orthogonally-adjacent cell that is joined to your current cell by
a declared corridor, that you have never visited before, and that you are
currently allowed to enter. Once you leave a cell you can never return to it
— your own trail permanently blocks the way back. The walk ends the moment
you have no legal move left (or you may stop earlier voluntarily). Collect
as much total reward as you can before that happens.

## Setup

The world is `V` cells, each at a distinct grid coordinate `(r, c)` with
`0 <= r, c < N`, and `E` corridor edges — **only** the listed edges are
usable; two orthogonally-adjacent cells with no listed edge between them
cannot be crossed. Each cell carries an integer `reward >= 0`, collected once
the first (and only) time you visit it. Some cells are **key** cells: visiting
one grants you its key id, permanently, for the rest of the walk. Some cells
are **gate** cells, each tagged with a required key id — a gate cell may only
be entered if you already hold the matching key from an *earlier* step. A
gate you cannot yet open behaves like a wall until you have the key.

## Input (stdin)

```
N V E S
r_1 c_1 reward_1 kind_1 keyid_1      (V lines, kind: 0=normal, 1=key, 2=gate)
...
r_V c_V reward_V kind_V keyid_V
u_1 v_1                              (E lines, 0-indexed corridor edges into the cell list above)
...
u_E v_E
```
`S` is the 0-indexed starting cell (its own reward is collected immediately).
`20 <= V <= 320`, time limit 5s.

## Output (stdout)

```
L
r_1 c_1                              (L lines: your visited cells, in walk order)
...
r_L c_L
```
The first cell must equal the start cell's coordinates.

## Feasibility

`1 <= L <= V`; every printed `(r,c)` must be a declared cell; all `L` cells
must be pairwise **distinct**; every consecutive pair must be joined by a
declared corridor edge; you may never enter a gate cell before its key was
collected on a strictly earlier step. Any violation scores `0`.

## Objective & Scoring

Maximise `F` = the sum of rewards over your (distinct) visited cells. The
checker also computes `B`, the reward collected by its own reference walker:
a fixed rule that, from any cell, always takes the legal move in priority
order **right, then down, then up, then left** among corridors it hasn't
used, regardless of reward size. Your score is
```
Ratio = min(1.0, 0.1 * F / B)
```
so that reference walker itself scores `~0.1`.

## What makes it hard

Corridors branch. Some branches are short **dead ends** whose entrance cell
looks generously rewarding — but once you step in, you can never come back
out, and the rest of the network (often far more valuable) is lost for good.
Elsewhere the corridor **forks** into two paths that later reconverge: one
fork is short and slightly richer per step but carries no key; the other is
a little longer, a little leaner, but the *only* place a certain key sits.
Beyond the point where the forks reunite sits a locked gate; only that key
opens it, and only a long, rewarding stretch beyond the gate makes the whole
trip worthwhile. Picking branches purely by their nearest reward — instead of
by what they preserve access to — throws away the vault forever, either by
dead-ending early or by starving at a gate you can no longer open.

## Example scoring

Suppose the walk collects `F = 620` and the reference walker collects
`B = 130`. Then `Ratio = min(1.0, 0.1 * 620 / 130) = min(1.0, 0.477) = 0.477`.

## Constraints

Time limit 5s, memory 512MB. `V <= 320`, `E <= 400`. Scoring is fully
deterministic.
