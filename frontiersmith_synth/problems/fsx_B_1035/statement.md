# Echo Chains Over the Villages of F_p*

## Problem

A prime `p` defines `p-1` villages, numbered `1, 2, ..., p-1` (the nonzero
residues mod `p`). A set `T` of villages needs to be lit tonight. You light
villages with **echo chains**. A chain is a triple `(a, r, L)` of integers
with `1 <= a, r <= p-1` and `L >= 1`: starting from anchor `a`, it echoes with
ratio `r`, lighting every village in

```
{ a mod p, a*r mod p, a*r^2 mod p, ..., a*r^(L-1) mod p }
```

(all arithmetic mod `p`). A chain of length `L` costs `L` units of relay
effort, and every chain you deploy — regardless of its length — also costs a
fixed setup fee `alpha` (given in the input) for building its transmitter.
Lighting a village outside `T` is harmless (no penalty, no credit) — only
`T` must end up fully lit. You may deploy up to `6000` chains with lengths
summing to at most `3,000,000` (far beyond what any sane construction
needs; the caps just bound the checker's work).

Useful fact: every ratio `r` has a multiplicative order `d | p-1` (smallest
`d>=1` with `r^d ≡ 1 mod p`); a chain with `L = d` closes into a cycle back
at its anchor, so going further only repeats villages already lit. Choosing
`r` and `L` well, relative to how `T` is laid out, is up to you.

## Input (stdin)

```
p alpha
m
t_1 t_2 ... t_m
```
`p` is prime, `2 <= alpha <= p`, `1 <= m <= p-1`, and `t_1 < ... < t_m` are
the (distinct) target villages, each in `[1, p-1]`.

## Output (stdout)

```
c
a_1 r_1 L_1
...
a_c r_c L_c
```
`c` is the number of chains you deploy (`1 <= c <= 6000`), followed by one
line per chain. Each of `a_i, r_i, L_i` must be an integer in `[1, p-1]`.

## Feasibility

Every listed `(a_i, r_i, L_i)` must satisfy `1 <= a_i, r_i, L_i <= p-1`.
`c` must be in `[1, 6000]` and `L_1+...+L_c` must not exceed `3,000,000`.
The union of the villages lit by all `c` chains must be a superset of `T`.
Any violation (out-of-range values, too many chains/too much total length,
unparsable/non-finite tokens, trailing garbage, or `T` left incompletely
lit) scores `0`.

## Objective

Minimize the total cost:
```
cost = (L_1 + L_2 + ... + L_c) + alpha * c
```

## Scoring

The checker computes your feasible cost `F` and its own baseline cost `B`
(one singleton chain `(t, 1, 1)` per target village, i.e. `B = m*(1+alpha)`),
then reports
```
Ratio = min(1.0, 0.1 * B / F)
```
so matching the baseline exactly (`F = B`) scores `0.1`, and `F` ten times
smaller than `B` (or better) saturates the cap at `1.0`. Feasible-but-worse-
than-baseline outputs still score a small positive number below `0.1`;
infeasible outputs score `0`.

## Constraints

`p` is at most a few thousand; `m <= p-1`; time limit 5s; memory 512MB.

## Example (worked score)

Let `p = 29`, `alpha = 3`, `T = {1, 12, 17, 28}`. (Check: `12^1=12, 12^2=28,
12^3=17, 12^4=1 mod 29`, so `12` has order `4`, and `T` is exactly the cycle
it sweeps.)

Baseline: `B = 4*(1+3) = 16`.

Four singleton chains
```
4
1 1 1
12 1 1
17 1 1
28 1 1
```
cover `T` with `F = 4*(1+3) = 16 = B`, giving `Ratio = min(1.0, 0.1*16/16) =
0.1` (this singleton recipe always lands at exactly the calibrated
baseline score).

A single ratio-12 chain
```
1
1 12 4
```
also covers `T` exactly, with `F = 4 + 3 = 7`, giving the substantially
better `Ratio = min(1.0, 0.1*16/7) ≈ 0.229` — one chain instead of four,
because `12`'s order matches how `T` was laid out. On the real test data
`T` is far larger, split across several such clusters (some with a few
villages deliberately missing), plus scattered decoys — finding which
order(s) fit, and how many chains versus how much swept length to spend
mopping up the gaps within a cluster, is the puzzle.

*Illustrative note:* the formula above is the real, exact scoring formula
used on every test case — only the specific `p`, `alpha`, and `T` in each
test file are hidden until you read the input.
