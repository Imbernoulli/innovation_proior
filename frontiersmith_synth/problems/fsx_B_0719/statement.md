# The Spice Rack Relay

## Problem

A street vendor keeps `n` numbered spice jars in one long rack: a row of
`n` slots, one jar per slot. Across the day, `m` customers arrive in a
fixed, known order; the `i`-th customer asks for jar `s_i`. Serving that
customer costs exactly the jar's current 1-based position in the rack
(the vendor reaches past every jar in front of it).

Between customers the vendor may reorganize the rack, but reorganizing
is not free: the only move allowed is an **adjacent swap** of two
neighboring jars, and every such swap costs 1. The vendor has a hard
total **swap budget `K`** for the entire day — spend it however and
whenever you like across the day, but never exceed it. The rack's
**initial arrangement**, chosen before customer 1 arrives, is free.

Some jars are **correlated pairs**: regulars who buy jar `a` also tend
to buy jar `b` in the same visit or shortly after, and this pair's
business clusters tightly in time — it can dominate one stretch of the
day and then go quiet for the rest of it. The input tells you exactly
which jars are paired. All other jars are ordinary singles, requested
at a roughly steady background rate through the whole day.

**Objective (minimize):** the day's total cost = (sum over all `m`
customers of the jar's position when served) + (total swaps actually
used).

## Input (stdin)

```
n m K P
a_1 b_1
...
a_P b_P
s_1 s_2 ... s_m
```
`n` jars (ids `1..n`), `m` customers, swap budget `K`, `P` correlated
pairs (each `a_i, b_i` a jar-id pair), then the day's request sequence
`s_1..s_m` (each in `1..n`).

## Output (stdout)

```
p_1 p_2 ... p_n
T
c_1 j_1
...
c_T j_T
```
`p_1..p_n`: the initial permutation (jar in slot 1, slot 2, ...).
`T`: total swaps used (must satisfy `0 <= T <= K`).
Each `(c_k, j_k)` means: immediately before customer `c_k` is served,
swap the jars currently in slots `j_k` and `j_k+1` (`1 <= j_k <= n-1`).
The `T` events must be listed in **non-decreasing** order of `c_k`
(events tied on the same `c_k` are applied in the order listed).

## Feasibility

- `p_1..p_n` must be a permutation of `1..n`.
- Every `c_k` in `[1,m]`, every `j_k` in `[1,n-1]`.
- The `c_k` sequence must be non-decreasing.
- `T <= K`.
Any violation (including non-finite/garbage tokens) scores `Ratio: 0.0`.

## Scoring

The checker replays your script exactly against `s_1..s_m` and computes
your total cost `F` (position sum + `T`). It also computes `B`, the
cost of the laziest valid script: keep the jars in their input-given
numbering (`p_i = i`) and use zero swaps. Your score is
`100 * B / F`, capped at `1000`, printed as `Ratio: <score/1000>` on
the final line.

## Worked example (illustrative shape only)

`n=4, m=5, K=3, P=1`, pair `(1,2)`, sequence `3 4 1 2 1`. A trivial
script (`1 2 3 4`, no swaps) pays positions `3+4+1+2+1=11`, so `B=11`.
A script that starts `3 4 1 2` and swaps slots `(2,3)` once right
before customer 3 (turning the rack into `3 1 4 2`) pays positions
`1+2+2+4+2=11` plus the `1` swap `= 12` — slightly worse here; the
point is that whether a swap helps depends on what fires *after* it,
which the real instances make worth exploiting.

## Constraints

`1 <= n <= 300`, `1 <= m <= 7000`, `0 <= P <= n/2`, time limit 5s,
memory 512MB.
