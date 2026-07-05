# Polar Research Base: Footprint Selection

## Story

A survey drone has mapped a rectangular sector of an antarctic ice sheet onto an
`H x W` grid of cells. Each cell holds an **integer net terrain value**:

- large **positive** values are science hotspots (buried meteorites, subglacial
  lakes, ice cores worth drilling);
- mildly **negative** values are ordinary ice (a small levelling/build cost);
- strongly **negative** values are crevasses / pressure ridges you would avoid.

You must stake out the base **footprint**: a non-empty, **4-connected** set of
grid cells the station will occupy. Because the plateau is battered by wind, every
unit of the footprint's exposed **outer boundary** (an edge separating a chosen
cell from an unchosen cell, or from the sector border) must be insulated at a
fixed cost `perim_cost` per unit edge.

This is an AtCoder-heuristic-contest-style objective reframed as a **static,
deterministically scored offline instance**: choose a connected region to
maximize its net score. Selecting a maximum-weight *connected* region under a
boundary penalty is NP-hard, so there is no easy optimum — several search
strategies trade off differently.

## Objective (maximize)

For a chosen set `S` of cells:

```
net(S) = ( sum of grid[r][c] over (r,c) in S )
         - perim_cost * ( number of unit boundary edges of S )
```

A boundary edge is counted once for every side of a chosen cell that faces an
unchosen cell or the grid border. **Maximize `net(S)`.**

## Candidate contract (isolated program)

Your program is run in an isolated subprocess. It reads ONE JSON object (the
public instance) from stdin and writes ONE JSON object to stdout.

**Input (stdin):**
```json
{
  "name": "sector101",
  "H": 20,
  "W": 20,
  "perim_cost": 2,
  "grid": [[ -2, 11, -1, ... ],  ...]   // H rows x W columns of integers
}
```

**Output (stdout):**
```json
{ "cells": [[r0, c0], [r1, c1], ...] }
```

`cells` must be a **non-empty** list of **distinct**, in-range integer `[r, c]`
pairs (`0 <= r < H`, `0 <= c < W`) that together form a **single 4-connected
region**. Any violation — disconnected set, out-of-range or duplicate cell, wrong
shape, crash, timeout, or non-JSON output — scores `0.0` on that instance.

## Scoring

Deterministic; no wall-clock is used. Per instance the evaluator computes:

- `b` = weak baseline net = best single-cell footprint = `max(v) - 4*perim_cost`;
- `U` = loose upper bound = sum of all strictly-positive cell values (ignores both
  connectivity and the perimeter penalty, so it is generally unreachable);
- `c` = your footprint's `net(S)`.

Your instance score is
```
r = clamp( 0.1 + 0.9 * (c - b) / max(1e-9, U - b), 0, 1 ).
```

Reproducing the single-cell baseline scores about `0.1`; approaching the
unreachable positive-mass bound approaches `1.0`; a footprint worse than the
baseline scores below `0.1`. The final `Ratio` is the mean over all instances.

## Strategy hints

- **Baseline:** the single highest-value cell (`~0.1`).
- **Greedy region growing:** grow from the best cell, annexing the boundary cell
  of largest positive net gain `v - perim_cost*(4 - 2k)` (where `k` is how many
  already-chosen neighbours it touches) until nothing improves.
- **Multi-seed growth:** repeat greedy growth from every hotspot and keep the best.
- **Local search / annealing:** accept short mildly-negative "bridges" to link
  separate hotspot clusters, keeping the best net footprint seen.
