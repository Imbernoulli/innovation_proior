# Selene Base: Pressurized Footprint Layout

**Family:** heuristic-contest-offline (offline AtCoder-heuristic-style scoring) &nbsp;·&nbsp;
**Format:** B (isolated heuristic evaluation) &nbsp;·&nbsp; **Objective:** maximize

## Story

You are laying out a lunar research base on a surveyed landing zone. The site is a
discrete `N x N` grid of regolith tiles. Each tile `(r, c)` has an integer **net value**
`net[r][c]` — the science and resource yield of pressurizing that tile, **minus** the
excavation and radiation-shielding cost of clearing it. Rich ore veins, ice pockets and
instrument sites form a handful of compact **positive deposits**; the bare regolith
between them is net-**negative** (occupying it still costs mass but returns nothing).

The base is a *single pressurized, walkable footprint*. You choose the set `S` of
occupied tiles subject to two hard rules:

- **Connectivity:** `S` must be **4-connected** — every occupied tile reachable from
  every other through orthogonally adjacent occupied tiles (no isolated domes).
- **Budget:** life-support caps the build at **at most `K`** occupied tiles.

You **maximize the total net value** of the footprint:

```
value(S) = sum of net[r][c] over all tiles (r,c) in S
```

The tension is coverage vs. connectivity vs. budget: the richest deposits sit apart,
separated by net-negative regolith moats. Taking one deposit is easy; stitching two rich
deposits together means spending scarce budget on negative **bridge tiles**, which only
pays off if the second deposit is rich enough. There is no easy optimum — this is a
budgeted maximum-weight connected-subgraph instance.

## Your program (isolated stdin → stdout)

You write a standalone program. It reads **one** JSON object (the PUBLIC instance) from
stdin and writes **one** JSON object (your footprint) to stdout. It is run in a fresh,
OS-sandboxed subprocess and only ever sees the public instance.

### Input (stdin) — one JSON object
```json
{
  "name": "site101",
  "n": 14,                     // grid is N x N
  "k": 18,                     // occupy at most K tiles
  "net": [[ ... ], ...]        // N rows of N integers (may be negative)
}
```

### Output (stdout) — one JSON object
```json
{"cells": [[r0, c0], [r1, c1], ...]}
```
`cells` is your footprint `S`: a list of `[r, c]` tile coordinates.

## Validity

A footprint is **valid** iff:

- `cells` is a list of **between 1 and K** pairs of integers,
- every coordinate satisfies `0 <= r, c < N`,
- all tiles are **pairwise distinct**, and
- the tiles form a **single 4-connected region**.

Wrong shape, out-of-range or duplicate tiles, a disconnected set, an empty set, more than
`K` tiles, a crash, a timeout, or non-JSON output → that instance scores **0.0**.

## Scoring (deterministic; no wall-time)

For each instance the evaluator computes two references itself:

- `v_base` = the value of the single **best tile** (max `net`). A one-tile footprint is
  always connected and within budget, so this is the weak baseline.
- `v_ub` = the sum of **all strictly-positive** tiles, ignoring connectivity and the
  budget `K`. A loose, generally unreachable upper bound — no single `K`-budget connected
  footprint can gather every scattered positive tile, leaving headroom below 1.0.

Your footprint value `v_cand = value(S)` is normalized with an affine anchor
(single-best-tile → 0.1, all-positive ideal → 1.0):

```
r = clamp( 0.1 + 0.9 * (v_cand - v_base) / max(1e-9, v_ub - v_base),  0,  1 )
```

Reproducing the single best tile scores ≈ 0.1; a footprint that dips net-negative scores
below 0.1; growing a connected region that spans rich deposits scores higher, capped at
1.0. The reported **Ratio** is the mean of `r` over all instances (a fixed, seeded family
of 12 sites, including larger held-out ones with more deposits than the budget can span).

## Strategy ladder (for reference)

- **Single best tile** — the weak baseline (≈ 0.1).
- **Single-deposit region growing** — seed at the best tile, annex the best positive
  frontier tile until the budget runs out or the frontier turns net-negative; captures
  one whole deposit but never bridges.
- **Multi-seed growing + budget-aware bridging + pruning** — grow from several rich
  seeds, connect additional deposits through the cheapest negative-tile bridge when it
  pays off, and prune negative leaf tiles.
- **Restart / local search** over which deposits to connect and how to spend the bridge
  budget.
