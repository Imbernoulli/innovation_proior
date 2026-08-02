# Cache-Cooperative Tiling: Tile, Pad, Prefetch

## Problem
You must schedule the canonical dense matrix product `C[i][j] += A[i][k]*B[k][j]`
for `i,j,k` in `[0,N)`, over an idealized memory system: a set-associative,
LRU, word-addressed cache of capacity `C` words, line size `L` words and
associativity `A` (so there are `S = C/(L*A)` cache sets). `A`, `B`, `C` are
each stored row-major, `N` words per logical row, but you may pad every row
of an array by up to `PAD_MAX` extra (unused) words — padding shifts which
cache set a row's addresses land in without changing which values are read.

You output a **tiling plan**: a cube tile size and an inner loop order. The
computation is tiled on all three axes with `range(0,N,T)`-style blocks (the
last block per axis is clipped to `N`, so `[0,N)` is always covered exactly,
whatever `T` is). Blocks are visited in a **fixed** `I,K,J`-major order (not
yours to choose); within the selected block you choose the scalar visitation
`inner_order` (a permutation of `i,j,k`). To force genuine tiling — not a
single giant block wearing a tiling costume — every tile size must satisfy
`min_t <= T <= max_t` where `max_t = floor(N/3)` (at least 3 blocks per axis)
and `min_t = min(2, max_t)` (a block must span more than one element). Because
every legal plan visits every `(i,j,k)` triple exactly once, the arithmetic
result is always exactly `C = A*B` regardless of tile size/padding/order —
your artifact is scored purely on how memory-friendly its address trace is,
not on correctness.

Every innermost iteration issues three memory events, in order: read
`A[i][k]`, read `B[k][j]`, read-modify-write `C[i][j]` (word address =
`base_array + row_index*(N+pad_array) + col_index`, arrays laid out back to
back). Whenever an event's cache line is exactly one more than the previous
event **on the same array's stream**, the simulator also silently prefetches
the *next* line of that stream into the cache (a free insert, not scored as
a hit or a miss) — sequential (stride-1) access patterns get help for free;
strided ones do not.

## Input (stdin)
```
N
C L A
PAD_MAX
```
`8 <= N <= 40`. `C, L, A` are positive integers with `L*A` dividing `C`
exactly. `0 <= PAD_MAX <= N`.

## Output (stdout)
```
Ti Tj Tk
padA padB padC
inner_order
```
`Ti,Tj,Tk` are integers in `[min_t, max_t]` (see above, computed from `N`).
`padA,padB,padC` are integers in `[0,PAD_MAX]`. `inner_order` is a
3-character permutation of `ijk`. Exactly these 7 whitespace-separated
tokens, nothing else — any extra/missing token, any non-plain-integer token
(no decimals, no scientific notation, no `nan`/`inf`), an out-of-range
value, or a malformed order string scores `Ratio: 0.0`.

## Objective
Minimize the total number of cache misses over the `3*N^3` memory events
induced by your plan.

## Scoring
Let `B` be the miss count of the fixed simplest-legal reference plan
(`Ti=Tj=Tk=min_t`, all padding `0`, `inner_order=ijk` — tiling with no
capacity, padding or order reasoning behind it). Let `F` be your plan's
miss count. The checker computes
```
Ratio = min(1.0, 0.1 * B / F)
```
The reference plan itself always scores exactly `0.1`; a 10x reduction in
misses relative to `B` reaches the cap of `1.0`.

## Constraints
- `8 <= N <= 40`, `2 <= L`, `1 <= A`, `C = L*A*S` for some integer `S >= 1`.
- Deterministic simulation; checker runs in `O(N^3)` and completes well
  within the time limit for every case.

## Example
`N=24`, `C=192` words, `L=4`, `A=48` (so `S=1`, a fully-associative cache).
Here `min_t=2`, `max_t=8`. The reference plan (`T=2`, no padding, canonical
order) gets `B=1862` misses. A plan that fills the cache capacity
(`Ti=Tj=Tk=8`, no padding, canonical `ijk` order) gets `F=1282` misses,
scoring `min(1.0, 0.1*1862/1282) = 0.145`. Co-designing the same tile size
with a small row padding and a prefetch-friendly inner order can drive `F`
lower still — and on caches with low associativity, where the naive tile's
rows alias into the same few sets, that padding stops being optional: the
capacity-only recipe can thrash far below even the reference plan while a
padded, order-aware plan cuts misses several-fold.
