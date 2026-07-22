# Boot Arena: Scheduling a Firmware Heap

## Problem
A firmware boot sequence needs `N` scratch buffers, `1..N`, buffer `i` sized `s_i` bytes.
They form an **alloc/free precedence DAG**: an edge `p -> c` means buffer `c` is computed
*from* buffer `p`, so `p` must already be allocated and must still be alive (not yet freed)
the instant `c` is allocated:
```
alloc(p) < alloc(c) < free(p)
```
Every buffer is also allocated exactly once and freed exactly once, with `alloc(i) <
free(i)`. Edges always point from a lower index to a higher index, so processing indices
`1..N` in order is one valid schedule -- but far from the only one: whenever two buffers
are unrelated in the DAG you may freely choose which to allocate or free first, and you may
delay a free well past the earliest legal moment if that helps.

Choose ONE full linear order of all `2N` alloc/free events (respecting every precedence
constraint above) that minimizes the **peak memory footprint** once it is replayed through
the firmware's fixed heap allocator.

## The allocator (fixed, deterministic)
The heap starts empty; a running "top" marks how far the arena has ever had to grow.
- **Alloc** `s`: scan free holes from the lowest address; place the block in the first hole
  large enough (**first-fit**), splitting off any leftover. If no hole fits, place at `top`
  and advance `top` by `s`.
- **Free**: return the block to the free list and **coalesce** it with any adjacent free
  hole. If the (possibly just-merged) hole now touches `top`, retract `top` to the hole's
  start (a classic "wilderness chunk" reclaim).

The **objective** `F` is the highest address ever reached (the historical high-water mark)
-- *not* the final size, and *not* the live-byte count at any instant. `F` depends on the
exact shape and timing of the holes your order creates; freeing eagerly is not, by itself,
enough.

## Input (stdin)
```
N
s_1 s_2 ... s_N
M
p_1 c_1
...
p_M c_M
```
`1 <= N <= 2000`, each `1 <= s_i <= 10^6`, `0 <= M <= 4*N`, every edge satisfies `p_i < c_i`.

## Output (stdout)
`2N` nonzero integers (any whitespace layout), each with absolute value in `[1,N]`: a
positive value `i` means "allocate buffer `i`", a negative value `-i` means "free buffer
`i`". Every buffer must appear as exactly one positive and one negative token.

## Feasibility
Scores `0` on: wrong token count; a non-integer, zero, or out-of-range token; a buffer
allocated more than once, freed more than once, freed before being allocated, or never
allocated/freed; or any DAG edge with `alloc(p) < alloc(c) < free(p)` violated.

## Scoring
Let `B = sum(s_i)` -- the cost of the trivial "allocate everything, free nothing until the
very end" schedule (always feasible, reuses no hole, `F = B` exactly for it). With your
achieved peak `F`:
```
Ratio = min(1.0, 0.1 * B / F)
```
Halving `F` doubles the score; a `10x` tighter packing caps at `1.0`. The optimal schedule
is not known in closed form -- interleaving buffers can sometimes still help when hole
sizes happen to line up, so there is genuine room above any fixed strategy.

## Constraints
Deterministic integer simulation only. Time limit 5 s, memory 512 MB.

## Example
`N=3`, sizes `10, 5, 5`, edges `1->2`, `1->3` (buffer 1 must stay live through both
children's allocations). Submit `1 2 -2 3 -3 -1`: alloc 1 (top 0->10); alloc 2 (no hole,
top 10->15, peak 15); free 2 (hole `[10,15)` touches top -> retracts to 10); alloc 3 (top
10->15 again); free 3 (retracts to 10); free 1 (hole `[0,10)` touches top -> retracts to
0). Peak `F=15`, baseline `B=20`, `Ratio = min(1, 0.1*20/15) = 0.133333`.

Both children were freed the instant they became eligible (earliest-legal, the obvious
heuristic) -- yet the peak is `3/2` the largest single buffer, since both children were
momentarily resident *alongside* buffer 1. Two effects compound in larger instances: (a) if
several unrelated buffer-groups are interleaved in the input, keeping every group alive at
once (instead of finishing one group before starting the next) makes the peak grow with the
*number* of groups, not just their sizes; (b) even a single, fully isolated chain of
dependent buffers can suffer -- freeing a buffer the instant it is legal can let first-fit
wedge a smaller, soon-needed buffer into part of its hole, leaving a sliver too small for a
same-size buffer that needs the *whole* hole shortly after. Freeing early does not fix
either effect; reordering *which* group runs when, and *when exactly* (not just *how soon*)
each buffer is freed, does.
