# The Endless Atrium

## Problem

You are designing the floorplan of a museum built on a `W x H` grid of unit rooms. You must
choose exactly `n` cells to open as gallery rooms. Two open rooms are adjacent if they share
a grid edge; the open rooms must form one connected complex (a visitor can walk from any open
room to any other through open rooms only). For accessibility, the fire code caps the
**graph diameter** of this complex: the largest number of doorway-crossings needed to walk
between the two open rooms that are hardest to reach from each other must be at most `D`
(graph-hop distance, not Euclidean distance).

Subject to that, you want visitors to *wander as long as possible before the layout feels
"mixed"* -- formally, you want to maximize the relaxation time of the following process. A
lazy wanderer standing in an open room stays put with probability 1/2; otherwise they step
through a uniformly random doorway to an open neighboring room (probability `1/(2*deg)` for
each of the room's `deg` open neighbors). This is a reversible Markov chain. Its transition
operator has the same eigenvalues as the symmetric matrix `S = 0.5*I + 0.5*D^-1/2 A D^-1/2`,
where `A` is the 0/1 adjacency matrix of your open rooms and `D` here is the diagonal degree
matrix (do not confuse with the diameter cap `D` above). Let `lambda_2` be the second-largest
eigenvalue of `S` (the largest is always 1). The **spectral gap** is `gap = 1 - lambda_2`, and
your score is driven by the **relaxation time** `F = 1/gap`: bigger means visitors take longer
to "forget" where they started, i.e. the floorplan mixes slowly.

Intuition, not a proof: a long empty corridor mixes slowly because it is *far end-to-end*, but
the diameter cap directly limits how long a corridor can be. A floorplan can still mix very
slowly with a *short* diameter if some single doorway is a severe bottleneck relative to the
room-volume on each side of it -- this is about **conductance** (cut-size vs. volume), not
distance, and the diameter cap does not forbid it.

## Input (stdin)

One line: `W H n D` -- grid width, grid height, number of rooms to open, diameter cap.

## Output (stdout)

Exactly `n` lines, each `r c` (0-indexed, `0 <= r < H`, `0 <= c < W`): the open rooms. All
`n` cells must be distinct.

## Feasibility

- Exactly `n` distinct, in-bounds integer cell coordinates.
- The induced 4-adjacency graph on your `n` cells must be **connected**.
- Its graph diameter (max shortest-path hop count over all pairs of open cells) must be `<= D`.
Any violation scores `Ratio: 0.0`.

## Scoring

The checker computes your relaxation time `F` as defined above via an exact, deterministic
dense eigendecomposition. It also builds its own simple reference floorplan `B` (a compact
near-square block of `n` rooms) and its relaxation time. Your score is
`Ratio = min(1000, 100 * F / B) / 1000`, so exactly reproducing the reference scores ~0.1, and
mixing 10x slower than the reference saturates the score at 1.0.

## Constraints

- `4 <= n <= 60`, grid sized so a compact block of `n` cells always fits with room to spare.
- `D` is always at least the diameter of the checker's own reference block (so a valid,
  if unambitious, solution always exists).
- Time limit 5s, memory 512MB.

## Example (worked score, illustrative FORM only)

For `n=4`: a straight line of 4 rooms `(0,0)-(0,1)-(0,2)-(0,3)` has diameter 3 and relaxation
time `F=4.0`. A 2x2 block `(0,0),(0,1),(1,0),(1,1)` has diameter 2 and relaxation time `F=2.0`
-- this block is also the checker's reference, so `B=2.0`. If the diameter cap allows `D>=3`,
submitting the line scores `min(1000, 100*4.0/2.0)/1000 = 0.2`, beating the reference (`0.1`)
because it mixes twice as slowly. At larger `n` with a tight `D`, a single long corridor
cannot hold all `n` rooms at all -- and simply thickening a short corridor with leftover rooms
(the obvious first idea) is *not* the best use of the spare rooms either.
