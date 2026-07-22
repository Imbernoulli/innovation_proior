# Composing a Musical Touch (Change-Ringing Blueline)

## Problem

You are composing a *touch* for `n` tower bells (`n` is always **odd**). Bells occupy
positions `1..n` in a *row*; a row is a permutation of the bell numbers. Ringing starts
and must end at **rounds**, the identity row `(1,2,...,n)`.

At each step the ringers apply a **call**: a set of pairwise non-overlapping adjacent
position-pairs `(i,i+1)`. Formally a call is a set of indices `S ⊆ {1,...,n-1}` such that
no two chosen indices differ by exactly 1 (so the swaps they trigger never touch a
shared bell). Applying a call swaps, simultaneously, the bells standing at positions `i`
and `i+1` for every `i ∈ S`.

A touch of length `K` is a sequence of `K` calls `c_1,...,c_K`, turning row `row_0 =
rounds` into `row_1, row_2, ..., row_K` by successive application. Two structural rules
(**trueness**) must hold:

1. **Closure**: `row_K` must again equal rounds.
2. **No-repeat**: the rows `row_0, row_1, ..., row_{K-1}` must be **pairwise distinct**
   (rounds itself may legitimately reappear only at the very end, as `row_K`).

## Input (stdin)

```
n Kmax PAL_BONUS
B
w_1 p_1,1 p_1,2 ... p_1,n
...
w_B p_B,1 p_B,2 ... p_B,n
```
`n` (odd, 5≤n≤9), `Kmax` (max touch length), `PAL_BONUS` (float, ≥0). Then `B` "musical"
target rows, each an integer weight `w_b` followed by a permutation of `1..n`.

## Output (stdout)

```
K
S_1
S_2
...
S_K
```
`K` on the first line, then `K` lines, each the call `S_t` as space-separated, strictly
increasing indices in `1..n-1` (no two differing by 1). `1 ≤ K ≤ Kmax`.

## Feasibility

Reject (score 0) if: any call line is malformed, contains a non-integer token, an index
outside `1..n-1`, two indices differing by less than 2, or is empty; `K` is out of range
`[1,Kmax]`; `row_K` is not rounds; or `row_0,...,row_{K-1}` are not pairwise distinct.

## Scoring

Define the **mirror** of a row `r` (an array indexed `1..n`): reverse the position order
and complement every bell number, `mirror(r)[i] = (n+1) - r[n+1-i]`. Rounds is a fixed
point of this map.

Your raw score is
```
F = K                                                                (length)
  + sum of w_b over musical rows that occur among row_1..row_{K-1}   (each row counted once)
  + PAL_BONUS * P
```
where `P` counts positions `t` in `1..K-1` with `row_t == mirror(row_{K-t})` — how much
of the touch's interior is symmetric under this position-reversing, bell-complementing
reflection. `P` is a genuine bonus, not a requirement: an asymmetric touch is perfectly
legal, it simply forfeits this term.

The checker builds its own small reference touch (an "out and back": call `{1}` then
call `{1}` again, `K=2`) to get a baseline value `B` for `F`, then reports
```
Ratio = min(1000.0, 100.0 * F / max(1e-9, B)) / 1000.0
```

## Example (worked, n=3, illustrative shape only — not a real test case)

`n=3`, rounds `=(1,2,3)`. Call `{1}` swaps positions 1,2: rounds → `(2,1,3)`. Call `{2}`
then swaps positions 2,3: `(2,1,3) → (2,3,1)`. Continuing with calls `{1},{2},{1},{2},{1}`
(6 calls total) returns to `(1,2,3)`, visiting all `6` distinct rows of `S_3` before
closing. Here `mirror((2,1,3)) = (1,3,2)`; if `row_5=(1,3,2)` then position `t=1`
contributes to `P` because `row_1=(2,1,3)=mirror(row_5)=mirror(row_{6-1})`.

## Constraints
`5 ≤ n ≤ 9` (odd), `1 ≤ B ≤ 10`, time limit 5s, memory 512MB, each `.in ≤ 5MB`.
