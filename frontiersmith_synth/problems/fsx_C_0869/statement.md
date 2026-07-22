# Shared-Hub Tolerance Budgeting

## Problem

A product has `m` interchangeable-quality features, numbered `0..m-1`. Each feature `f`
is assigned a tolerance **grade** `g(f)` in `{0,1,...,6}`. Higher grade = finer
manufacturing precision. Two fixed tables (identical for every instance) convert a
grade into a **cost** (what you pay) and a **tolerance value** (how much slack the
feature contributes to a spec check — lower is better):

```
grade g :   0   1   2   3   4    5    6
cost(g) :   0   1   3   7  15   31   63     (= 2^g - 1)
tol(g)  :  64  32  16   8   4    2    1     (= 2^(6-g))
```

There are `C` requirement **chains**. Chain `i` names three features — two
**private** features `p1, p2` used by no other chain, and one **hub** feature `h`
that may be reused by many chains — plus two integer spec bounds `spec_primary` and
`spec_backup`. The chain imposes two checks that must BOTH hold:

- **PRIMARY**: `tol(g(p1)) + tol(g(p2)) + tol(g(h)) <= spec_primary`
- **BACKUP** (worst case): `tol(g(p1)) + tol(g(p2)) <= spec_backup` — this must hold
  using only the chain's own private features, as if the shared hub were degraded or
  unavailable; it is checked unconditionally, not only when the hub actually fails.

Many chains typically share the same hub feature — a hub touched by many chains has
high "betweenness" in the requirement network. Chains do not share private features.

## Input (stdin)

```
m C
p1_0 p2_0 h_0 spec_primary_0 spec_backup_0
...
p1_{C-1} p2_{C-1} h_{C-1} spec_primary_{C-1} spec_backup_{C-1}
```
All values are non-negative integers, `0 <= p1,p2,h < m`. `m` and `C` scale up to a
few hundred across the 10 test cases.

## Output (stdout)

Exactly `m` whitespace-separated integers: `g(0) g(1) ... g(m-1)`, each in `[0,6]`.

## Feasibility

Every one of the `C` chains' PRIMARY and BACKUP checks must hold simultaneously under
the SAME grade assignment. Any violation, any malformed/incomplete output, or any
out-of-range grade scores `0`.

## Objective

Minimize the total cost `F = sum over all m features of cost(g(f))`.

## Scoring

The checker also builds its own naive reference assignment (never look at your
output): for every chain check, it pretends each of that check's members must
individually shoulder a `1/(3x member_count)` share of the spec bound (a fixed safety
multiplier of 3), and grades every feature to the tightest such requirement across all
checks it appears in — this ignores that a heavily-shared hub feature might be worth
disproportionate investment. Call its cost `B`. Your score is
`min(1000, 100*B/F) / 1000`, i.e. matching the naive reference scores `0.1`; costing
10x less scores `1.0`. Beating it by exploiting sharing scores in between.

## Constraints

`1 <= m <= 800`, `1 <= C <= 400`. Time limit 5s, memory 512MB.

## Example (worked score)

Instance: `m=13 C=6`; six chains `(2i, 2i+1, 12, 29, 28)` for `i=0..5` all share hub
feature 12.

Solved chain-by-chain (each chain sees only its own `spec=29/28`): the cost-optimal
split for ANY one of these chains alone is private grades `(3,3)` with hub grade `3`
— tol `8+8+8=24<=29` (primary), `8+8=16<=28` (backup), cost `7+7+7=21`. Merging six
identical per-chain answers by taking the per-feature max leaves hub at grade `3` too:
`F = cost(3)*13 = 91`.

Recognizing the hub is shared by all six chains changes the calculus: raising the hub
one grade to `4` (tol `4`, cost `15`, paid ONCE) widens EVERY chain's remaining
primary budget from `29-8=21` to `29-4=25`, letting each chain's privates drop to
`(2,3)` (tol `16+8=24<=25`, cost `3+7=10`) instead of `(3,3)` (cost `14`) — a saving
of `4` per chain, `24` total, against a one-time hub cost increase of `15-7=8`. Net:
`F = cost(4) + 6*10 = 15+60=75`, `14%` cheaper than the per-chain-independent answer,
because the hub investment paid for itself six times over instead of once.
