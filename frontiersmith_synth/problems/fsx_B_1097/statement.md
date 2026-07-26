# Marker-Ledger Layering: Weighted Subsequence Antichains

## Problem
A *ledger code* is a nonempty string over the marker alphabet `{0, 1, ..., a-1}` with
length between `1` and `Lmax`. Code `x` **endorses** code `y` (with `x != y`) if `x`
occurs as a **scattered subsequence** of `y` -- the characters of `x` appear inside `y`
in order, not necessarily contiguously (e.g. `"01"` endorses `"201"` and `"0011"`, but
not `"10"`).

You submit a **set** of ledger codes that is an **antichain**: no code in the set may
endorse another code in the set (equal-length distinct codes automatically satisfy
this, since neither can be a proper subsequence of the other -- only codes of
*different* lengths can create a conflict).

Two extra ledger rules apply, given in the input:
- **Per-length ceiling** `cap[l]`: at most `cap[l]` of your kept codes may have length
  `l`.
- **Global budget** `T`: you may keep at most `T` codes in total.

Each kept code of length `l` contributes `weight[l]` to your score (weights are given
per length, so all codes of the same length are worth the same). Maximize the total
weight of the kept set.

A single length's ceiling is always far smaller than the number of codes of that
length that actually exist (`a**l`), so filling one length to its ceiling never uses
up the whole alphabet. Codes built from **disjoint** digit subsets can never endorse
each other regardless of their lengths (a code using only digits from one subset
cannot possibly appear, in order, inside a code that never uses any of those digits) --
so several different lengths can sometimes be "stacked" together by giving each one
its own slice of the alphabet, rather than pouring every digit into a single length's
layer.

## Input (stdin)
```
a Lmax T
weight[1] weight[2] ... weight[Lmax]
cap[1] cap[2] ... cap[Lmax]
```
All values are positive integers; `2 <= a <= 4`, `1 <= Lmax <= 8`.

## Output (stdout)
```
K
code_1
code_2
...
code_K
```
Print the number of kept codes `K`, then the `K` codes (each a string of digit
characters, one per line, length between `1` and `Lmax`, digits in `0..a-1`).

## Feasibility
The output is valid iff **all** hold:
- `0 <= K <= T` and the header count matches the number of printed codes;
- every code has length in `[1, Lmax]` and uses only digits `0..a-1`;
- the `K` codes are pairwise distinct;
- for every length `l`, at most `cap[l]` of the kept codes have that length;
- no kept code endorses (is a scattered subsequence of) another kept code.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = sum over kept codes of weight[length(code)]`.

## Scoring
The checker builds its own reference set -- just the full length-1 layer, i.e. all
`min(cap[1], a, T)` single-digit codes -- worth `B = weight[1] * min(cap[1], a, T)`.
Your score is `Ratio = min(1.0, F / (10*B))`, so matching the checker's own
length-1-only baseline scores `0.1`; you need real structure to score higher.

## Constraints
`2 <= a <= 4`, `1 <= Lmax <= 8`, weights and caps are positive integers, `T` fits in a
32-bit integer. Time limit: 5s. Memory: 512MB.

## Example (worked score, illustrative only -- not a real test)
`a=4`, `Lmax=2`, `T=10`, `weight=[10,16]`, `cap=[4,6]`. Baseline: length-1 layer
`{"0","1","2","3"}`, `B = 10*4 = 40`.

Committing everything to the higher-weight length 2 (the obvious move) still hits
`cap[2]=6`, e.g. `{"00","01","02","03","10","11"}`: `F=6*16=96`, `Ratio=0.24`.
Length 2's ceiling (6) is far below its `a**2=16` possible codes, so most digits sat
unused. Reserving just **one** digit (`3`) for a length-1 code and the other
**three** digits `{0,1,2}` for length 2 still reaches length 2's ceiling
(`3**2=9>=6`) while adding one unconflicting length-1 code:
`{"3"} U {"00","01","02","10","11","12"}`, `F=10+96=106`, `Ratio=0.265`. The
length-1 code never uses digits `0/1/2`, so it cannot appear inside any length-2
code -- no endorsement either way. The real instances give enough spare alphabet
and enough distinct lengths for this layering to clearly beat committing to one
length's ceiling.
