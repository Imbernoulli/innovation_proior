# Collision-Spread Hash Offsets — a large progression-free set in Z_p

## Problem
A hash table with a prime number of buckets `p` reprobes a colliding key by adding a
fixed *offset* from a chosen offset set `S ⊆ {0,1,…,p-1}` (all arithmetic mod `p`).
A well-known pathology of linear/arithmetic reprobing is a **cascade collision**: three
offsets `a, b, c ∈ S` that sit in arithmetic progression modulo `p`

```
a + c ≡ 2·b   (mod p),   a ≠ c
```

let a single overloaded bucket relay its overflow straight down the chain, defeating the
spread. We therefore want an offset set that is **progression-free** (contains no such
triple) yet **as large as possible**, so the table has the widest possible menu of safe
reprobe steps.

Formally: given an odd prime `p`, construct a subset `S ⊆ {0,…,p-1}` such that there is
**no** triple of elements of `S` in 3-term arithmetic progression modulo `p`. Your score
grows with `|S|`. The exact maximum size of such a set in `Z_p` is not known in general —
this is an open, actively-studied quantity — so there is real headroom above any simple
construction.

## Input (stdin)
A single line with the odd prime `p`  (`23 ≤ p ≤ 200003`).

## Output (stdout)
The elements of your set `S`, as whitespace-separated integers (any layout — spaces and/or
newlines). Output the elements **only** (no leading count). Each element must be an integer
in `[0, p)`, and all elements must be distinct.

## Feasibility
An output is *feasible* iff:
- every token is an integer in `[0, p)` (non-integer / `nan` / `inf` / out-of-range → 0),
- no element is repeated, and
- `S` is progression-free mod `p`: there is no `a,c ∈ S` with `a ≠ c` whose midpoint
  `m ≡ (a+c)·2⁻¹ (mod p)` also lies in `S`.

Any violation scores exactly `0`.

## Objective (maximize)
`F = |S|`.

## Scoring
Deterministic. The checker builds its own baseline set `B` (the "no-carry" base-3 offset
set on a short register — a valid progression-free set it constructs internally) and reports

```
Ratio = min(1.0, 0.1 · |S| / |B|)
```

A set matching the internal baseline scores ≈ 0.1; you must roughly **10×** the baseline to
saturate at 1.0. The maximum is genuinely open, so full saturation is not expected.

## Constraints
- `23 ≤ p ≤ 200003`, `p` prime.
- Scoring is exact integer arithmetic; no timing, no randomness.

## Example
For `p = 23`, the set `S = {0, 1, 3, 4, 9, 10, 12, 13}` (base-3 numbers using only the
digits 0 and 1, all below `p/2`) is progression-free mod 23: for every pair its modular
midpoint falls outside `S`. It has `|S| = 8`. Whether a larger progression-free set exists
in `Z_23` — and how to find it for large `p` — is exactly the challenge.
