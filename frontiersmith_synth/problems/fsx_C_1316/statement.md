# Ring Substitution Pattern Under a Steric-Clash Correction and a Synthesis-Step Budget

## Problem
A scaffold has `N` substitution positions arranged in a ring (position `i`
is adjacent to positions `(i-1) mod N` and `(i+1) mod N`). You are given a
library of `K` candidate substituents; each has an electronic parameter,
a steric-bulk parameter, and a synthesis cost (in steps). You choose, for
every position, either to leave it unsubstituted (H) or to attach one
substituent type from the library (any type may be reused at multiple
positions). The total synthesis cost of everything you attach must not
exceed a given step budget.

A deterministic property surrogate scores your pattern. It is **additive**
across positions — each attached substituent contributes its electronic
term plus a steric bonus term — **except** for one correction: if two
BOTH-"bulky" substituents (steric bulk above a threshold given in the
input) end up ring-adjacent, they clash, and the penalty is large enough
to flip their *combined* steric contribution from positive to negative.
The additive model is a good guide to which substituents are worth their
cost in isolation, but it is blind to this pairwise interaction — and
ignoring it is costliest exactly on the substituent that looks best alone.

Your goal is to get the resulting property value as close as possible to
a given target.

## Input (stdin)
```
N K budget
P0 alpha beta s_thresh
target window
e_1 s_1 c_1
...
e_K s_K c_K
```
`N` ring positions (0-indexed), `K` substituent types (1-indexed in the
output). `budget` is the total synthesis-step cap. `P0` is the scaffold's
unsubstituted property value. `alpha` is the per-substituent steric bonus
coefficient; `beta` is the adjacency clash-penalty coefficient
(`beta > alpha`, so a clashing pair's net steric term is negative).
`s_thresh` is the steric-bulk value above which a substituent counts as
"bulky". `target` and `window` define the property you are aiming for.
For type `t`: `e_t` is its electronic term (any sign), `s_t` is its steric
bulk (non-negative), `c_t` is its synthesis cost (positive integer).

## Output (stdout)
Exactly `N` integers (any whitespace layout). Token `i` (0-indexed) is `0`
if position `i` is left unsubstituted, or `j` (`1..K`) if substituent type
`j` is attached there.

## Feasibility
Exactly `N` tokens, each a base-10 integer in `[0, K]`. The sum of `c_t`
over every attached substituent must be `<= budget`. Any wrong count,
non-integer token, out-of-range value, or budget overrun makes the whole
answer infeasible (score 0).

## Objective (what the score rewards)
Let `assign[i]` be your choice for position `i` (`-1` for unsubstituted,
`0..K-1` for a substituent index). The property value is:
```
S = sum over attached i of (e[assign[i]] + alpha * s[assign[i]])
  - sum over ring-adjacent pairs (i, i+1 mod N), both attached and both
    bulky (s > s_thresh), of beta * (s[assign[i]] + s[assign[i+1]])
P = P0 + S
```
The score rewards closeness of `P` to `target`, via
`closeness = 1 / (1 + |P - target| / window)` (closer is higher, capped
at 1 exactly on target).

## Scoring
The checker computes your `closeness` and the closeness of the "attach
nothing" pattern (`P = P0`, its own reference construction), then reports
`Ratio = min(1000, 100 * closeness / closeness_ref) / 1000`. Matching the
reference gives ≈0.1; the target is placed deterministically out of exact
reach, so no pattern reaches `closeness = 1`.

## Constraints
`6 <= N <= 16`, `4 <= K <= 8`, `1 <= c_t <= 4` (integer),
`0.4 <= alpha <= 0.7`, `1.3 <= beta <= 2.0`. Time limit 5s.

## Example (illustrative FORM only — not a real hidden case)
`N=4, K=2`, `budget=4`, `P0=10, alpha=0.5, beta=2.0, s_thresh=3.0`,
`target=14, window=2`. Library: type 1 `e=2 s=5 c=2` (bulky, high value),
type 2 `e=0.5 s=1 c=1` (mild, not bulky). Type 1 at positions 0 and 1
(ring-adjacent, both bulky): `S = (2+2.5)+(2+2.5) - 2.0*(5+5) = 9-20=-11`,
`P=-1`, far below target. Type 1 at positions 0 and 2 instead (NOT
ring-adjacent — no clash): `S = (2+2.5)+(2+2.5) = 9`, `P = 19`. Mixing in
type 2 to land nearer 14 does better still — WHERE the bulky group goes,
not just whether it fits the budget, decides the payoff.
