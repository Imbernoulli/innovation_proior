# Fixed-ROM Codeword Binding

## Problem
A hardware encoder's ROM layout was etched at fabrication time: it has `N` physical
codeword *slots*, each wired to a specific bit-length. Together the slot lengths
`L_1..L_N` form a valid **prefix-free code shape** (they are exactly the leaf depths
of some full binary tree, so `sum(2^-L_i) = 1`) — but the layout is fixed hardware,
not something you can redesign.

`N` distinct symbols will each appear in the stream exactly once as a *new* symbol,
in a fixed arrival order (symbol `i` is the `i`-th symbol to ever appear). The instance
tells you, up front, symbol `i`'s **final total occurrence count** `f_i`: the total
number of times symbol `i` occurs anywhere in the whole stream, counting its first
appearance itself as one of those occurrences — the complete future frequency
schedule is computable from the input before you bind anything.

As each symbol makes its first appearance you must **immediately and permanently bind
it to one still-unused ROM slot** (its codeword length becomes that slot's length).
Once bound, a symbol keeps that slot forever — there is no rebalancing, no swapping
a slot already given to an earlier symbol, and no slot may be reused.

Your artifact does not need to replay the online decision process — it just states
the **final binding**: for each symbol, in arrival order, which slot-length it ended
up with. Feasibility only requires that the final binding is achievable, i.e. that it
actually uses the given multiset of slot lengths, one slot per symbol.

## Input (stdin)
```
N
f_1 f_2 ... f_N
L_1 L_2 ... L_N
```
`f_i` is symbol `i`'s final total occurrence count (positive integer, symbol `i` = the
`i`-th symbol to first appear). `L_1..L_N` is the ROM's fixed slot-length multiset,
listed in an arbitrary (not frequency-sorted) order — the *set* of lengths is fixed
hardware, but which physical slot binds to which symbol is your decision. `1 <= L_i`,
`sum(2^-L_i) = 1` exactly (a genuine complete binary tree shape).

## Output (stdout)
`N` integers `d_1 d_2 ... d_N`: the slot length bound to symbol `i` (arrival order).

## Feasibility
The multiset `{d_1,...,d_N}` must equal the multiset `{L_1,...,L_N}` exactly (every
slot used exactly once). Any wrong count, non-integer/non-finite token, or multiset
mismatch scores `0`.

## Objective
Minimize the total encoded length over the whole stream:
```
F = sum_i f_i * d_i
```

## Scoring
Let `B` be the checker's own baseline: the "important symbols deserve the biggest
slot" binding — the highest-`f_i` symbol gets the *longest* codeword, the next-highest
the next-longest, and so on. By the rearrangement inequality this is provably the
*worst* possible binding of this exact slot multiset, so it is a safe normalizer.
With your feasible `F`:
```
ratio = min(1.0, 0.1 * B / F)
```
Reproducing that worst-case baseline scores `0.1`. Ties in `f` break by arrival index.

## Constraints
`8 <= N <= 200`, `1 <= f_i <= 2*10^6`, `1 <= L_i <= 120`, 10 test cases of increasing
size and (for several cases) an arrival order that is deliberately anti-correlated
with final frequency — the biggest-eventually symbols arrive last, after most of the
short slots are gone.

## Example (worked, illustrative shape only)
`N=4`, `f = [1, 1, 1, 100]`, `L = [1, 2, 3, 3]` (slots given in that order).
Worst-case baseline: symbol 4 (freq 100, the max) gets the longest slot (3), the
rest fill in descending: `F_worst = 1*1+1*2+1*3+100*3 = 306`, so `B=306`.
A binding that instead saves symbol 4 the shortest slot, e.g. `d = [2,3,3,1]`
(still a permutation of `{1,2,3,3}`, arrival order): `F = 1*2+1*3+1*3+100*1 = 108`.
`ratio = min(1.0, 0.1*306/108) = min(1.0, 0.283) = 0.283` — much better than `0.1`,
showing the binding matters even though every `f_i` was known from the start.
