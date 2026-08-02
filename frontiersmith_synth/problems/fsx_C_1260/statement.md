# Cosmic-Ray Layout: Interleave Cheap Codes Around Physical Bursts

## Problem
A memory row has `N` physical cells, indexed `0..N-1` left to right on the die. A cosmic
ray can strike a contiguous run of physically adjacent cells at once — a "multi-bit
upset" — flipping every cell in that run. You must partition the `N` cells into disjoint
groups ("codewords") and protect each codeword with an error-correcting code drawn from a
fixed catalog. A code with word length `w` and correction capability `t` can silently fix
up to `t` simultaneously-flipped cells **inside that one codeword**; if a single codeword
ever receives more than `t` flipped cells from one physical burst, the burst is
uncorrected. Decoding a codeword costs `cost` abstract ops (a decoder-latency surrogate) —
stronger codes decode more expensively. Cells may be assigned to codewords in ANY order —
grouping physically far-apart cells into the same codeword is allowed and free.

Your row must survive **every possible burst of every length from 1 up to `LMAX`**, at
every possible starting position (this is exhaustively re-checked, not sampled). Subject
to that, minimize the total decoding cost, summed over all codewords you use.

*Worked illustration (not the scored input):* `N=4`, one burst length `LMAX=2`, catalog
`{(w=2,t=1,cost=3), (w=4,t=2,cost=20)}`. Splitting into two codewords `{0,1}` and `{2,3}`
(word length 2, t=1) fails: the burst `{1,2}` puts one flipped cell in each codeword
(fine, t=1 each — OK actually), but the burst `{0,1}` puts 2 flipped cells in codeword
`{0,1}` (needs t≥2, have t=1) — infeasible. Interleaving instead — codeword A = `{0,2}`,
codeword B = `{1,3}` — spreads every length-2 burst across both codewords (≤1 cell each),
so `t=1` suffices: total cost `3+3=6`, versus the single `w=4,t=2` codeword costing `20`.

## Input (stdin)
```
N M LMAX
w_1 t_1 cost_1
...
w_M t_M cost_M
```
`M` catalog entries, 0-indexed `0..M-1` in the order given. `1 <= N <= 128`,
`1 <= LMAX <= N`, each `w_i` divides `N`, `1 <= t_i <= w_i`, `cost_i >= 1`.

## Output (stdout)
```
B
code_idx_1 cell cell ... cell   (w_1 cells)
...
code_idx_B cell cell ... cell   (w_B cells)
```
`B` codewords. Line `i` names a catalog index and lists exactly `w` cell indices for that
code's word length `w`. Every cell `0..N-1` must appear in **exactly one** line (a
partition of all `N` cells). Tokens may be split across lines freely.

## Feasibility
Reject (score 0) on: wrong/missing token counts, a cell index or catalog index out of
range, any cell listed twice or never listed, non-finite (`nan`/`inf`) or non-integer
tokens. Otherwise, for every burst length `Len` in `1..LMAX` and every starting offset
`s` in `0..N-Len`, let `Len` consecutive cells `s..s+Len-1` flip simultaneously; if any
codeword receives more flipped cells from that window than its `t`, the output is
infeasible (score 0).

## Objective
Minimize `F` = sum of `cost_i` over the `B` codewords you used (each codeword pays its
chosen code's cost once, regardless of word length).

## Scoring
The checker builds its own baseline `Bc`: contiguous (index-order) blocking using the
single largest catalog word length below `N`, with whichever correction strength that
plain blocking forces — a "no search, no interleaving" reference. With your total cost
`F`:
```
Ratio = min(1, 0.1 * Bc / F)
```
Smaller `F` (cheaper decoding, still fully correcting) scores higher.

## Constraints
`N <= 128`, `M <= 40`, `LMAX <= N`, time limit 5s, memory 512MB.
