# Stocking the Coset Workshop

## Problem
A workshop manufactures custom gears. Every commissioned gear is identified by a nonzero
value modulo a prime `p`; gears live in the multiplicative group `(Z/pZ)^*`, and "combining
parts" means multiplying mod `p`.

Before any orders are assembled you must **stock** the workshop: choose a table `S` of `m`
elements of `(Z/pZ)^*` ("master parts"). For each commissioned target value `t`, you must
**assemble** it as an explicit product of parts drawn from `S` (with repetition, and any
integer exponent, positive or negative):
`t ≡ s_{i_1}^{e_1} · s_{i_2}^{e_2} · ... · s_{i_k}^{e_k} (mod p)`.
Every reference to a stocked part — regardless of its exponent — counts as **one factor** in
that order's assembly recipe.

You see all `T` commissions in advance, but you don't know in advance which of them share
hidden structure and which are awkward one-off decoys planted by a difficult client. Stocking
more parts (`m`) is pure overhead; so is a longer per-order recipe. Both cost you, and your
job is to minimize the total.

## Input (stdin)
```
p
g
LAMBDA
T
t_1 t_2 ... t_T
```
- `p` is prime; `g` is a primitive root of `p` (a generator of the full group `(Z/pZ)^*`),
  given for convenience.
- `LAMBDA` is the per-stocked-part overhead cost (a positive integer).
- `T` is the number of commissions; `t_1, ..., t_T` are targets, each in `[1, p-1]`.
- **Guarantee**: `t_1` is a "clean" commission — it lies in the same hidden coset as most
  (but not all) of the other targets. The rest of the `T-1` targets are an unlabeled mix of
  further clean commissions from that same coset, and a handful of decoys that are not.

## Output (stdout)
```
m
s_1 s_2 ... s_m
k_1 idx_1 e_1 idx_2 e_2 ... idx_{k_1} e_{k_1}
...
k_T idx_1 e_1 ... idx_{k_T} e_{k_T}
```
`m` is your stock size; `s_1..s_m` are the stocked parts (each in `[1, p-1]`). Then, one line
per target in input order: `k_i` (number of factors used for that target), followed by `k_i`
pairs `(idx, e)` where `idx` is a 1-based index into your table and `e` is any integer exponent
with `|e| <= 10^7`. The claimed product must equal `t_i (mod p)` exactly.

## Feasibility
Score `0.0` if any of: some `s_j` lies outside `[1,p-1]`; some `idx` lies outside `[1,m]`;
some `k_i` lies outside `[1,500]`; `m` lies outside `[1,5000]`; a line is missing or
malformed; or, for any target, the claimed product `prod_j s_{idx_j}^{e_j} mod p` fails to
equal `t_i` exactly.

## Objective (minimize)
`F = (sum over all T targets of k_i) + LAMBDA * m` — total assembly work plus stocking
overhead.

## Scoring
The checker's own baseline stocks every target as its own dedicated part: `m = T`, one factor
each, so `B = T * (1 + LAMBDA)`. Then:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
A smarter stocking that needs much less total work than the dedicated-part baseline scores
well above `0.1`; a wasteful one scores at or below it.

## Example (worked, illustrative shape only — unrelated to the hidden structure above)
`p=13, g=2` (a primitive root of 13), `LAMBDA=2`, `T=3`, targets `= [2, 4, 8]`. One valid
stocking: `S=[2]` (`m=1`). Recipes: `1 1 1` (`2 = 2^1`), `1 1 2` (`4 = 2^2`), `1 1 3`
(`8 = 2^3`). Total factors `= 3`, so `F = 3 + 2*1 = 5`. Baseline `B = 3*(1+2) = 9`.
`sc = min(1000, 100*9/5) = 180`, so `Ratio: 0.180000`.

## Constraints
`3 <= p < 10^6`; `1 <= T <= 600`; `1 <= LAMBDA <= 100`; each `t_i, g` in `[1,p-1]`; time
limit 5s, memory 512MB. Each test input is well under 5MB.
