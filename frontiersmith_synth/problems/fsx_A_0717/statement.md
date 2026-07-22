# Cross-Family Mutant Kill Suite

## Problem
A reference program computes a piecewise-arithmetic function `R(x)` over integer
inputs `x` in `[0, D)`. Your QA team maintains **K = 6 mutant families** — each
family is a bug tribe an engineer might introduce (a wrong constant, a wrong
slope, a flipped sign, an off-by-one at a segment boundary, or a rarer logic
slip buried deep inside a segment's interior). Every family contains several
concrete **mutants**: buggy variants of `R`. For each mutant `m`, the input
already tells you the exact set of inputs on which `m` disagrees with `R` —
this **disagreement region** is always a single contiguous interval `[lo, hi)`
of integers (it may be as wide as a whole segment, or as narrow as one point).
A test input `x` **kills** mutant `m` iff `lo <= x < hi`.

You must design a tiny regression suite: at most `T` integer test inputs. A
family is "covered" in proportion to how many of ITS mutants get killed by
*some* input in your suite. Your suite's quality is the **worst** family's
coverage fraction — one neglected bug tribe drags the whole suite down, no
matter how thoroughly the other tribes were caught. Families are NOT equally
easy: some have wide, numerous disagreement regions that almost any spread of
tests will stumble into; others hide a single narrow interval deep inside one
segment, invisible unless you specifically aim for it. A strong suite reasons
about *where each family's disagreement regions are hardest to hit*, not about
which family happens to have the most mutants.

## Input (stdin)
```
D T K
n_0
lo hi          (n_0 lines)
n_1
lo hi          (n_1 lines)
...
n_{K-1}
lo hi          (n_{K-1} lines)
```
`D` is the input domain `[0, D)`. `T` is your test-point budget. `K = 6` is
the number of families, always listed in this fixed order: `const_shift`,
`coef_perturb`, `sign_flip`, `boundary_shift`, `interior_anomaly`,
`sparse_anomaly`. Family `f` has `n_f` mutants; each of the following `n_f`
lines gives one mutant's disagreement interval `[lo, hi)`, with
`0 <= lo < hi <= D`. Every family has at least one mutant.

## Output (stdout)
Print `c` (`0 <= c <= T`), the number of test inputs you choose, on the first
line, then `c` space-separated integers `x_1 ... x_c` (each `0 <= x_i < D`,
duplicates allowed but wasteful) on the second line. Print nothing else.

## Feasibility
Reject (score 0) unless: the output contains exactly `1 + c` integer tokens
total, `0 <= c <= T`, and every `x_i` satisfies `0 <= x_i < D`.

## Objective
For family `f`, let `killed(f)` be the number of its mutants whose interval
contains at least one submitted `x_i`. Maximize
```
F = min over f of ( killed(f) / n_f )
```

## Scoring
The checker builds its own trivial baseline suite `Bxs`: split the budget `T`
round-robin across the `K` families (`T // K` points each, remainder to the
first families), and within a family aim at the midpoints of a few
evenly-spaced mutants (no attempt to find the hardest family). Let `B` be this
baseline's own `F` value (family-minimum coverage), floored at `1e-9`. Then
```
sc    = min(1000.0, 100.0 * F / B)
Ratio = sc / 1000.0
```
Reproducing the baseline scores exactly `Ratio = 0.1`.

## Constraints
- `440 <= D <= 1700`, `6 <= T <= 9`, `K = 6`.
- Time limit 5s, memory 512MB. Each `.in` file is well under 5MB.
- All families and mutants are fully determined by the input; scoring is a
  pure deterministic function of your submitted integers.

## Example
Suppose `K=2` (illustrative shape only — the real input always has `K=6`),
`D=20, T=2`. Family 0 has mutants `[0,10)` and `[10,20)` (huge, easy). Family
1 has mutants `[3,4)` and `[16,17)` (single points, hard). The baseline picks
one point per family aimed at each family's first mutant's midpoint:
`x=4` (family 0) and `x=3` (family 1). That kills only mutant `[0,10)` and
only mutant `[3,4)`, so `killed=(1/2, 1/2)`, `B=0.5`, and the baseline itself
scores exactly `Ratio=0.1` by definition. The suite `{3, 16}` instead lands
`x=3` inside both `[0,10)` and `[3,4)`, and `x=16` inside both `[10,20)` and
`[16,17)` — every mutant in both families dies: `killed=(1,1)`, `F=1.0`, so
`Ratio = min(1000, 100*1.0/0.5)/1000 = 0.2`, double the baseline, from the
SAME budget of 2 points, just placed where the families' hardest (narrowest)
mutants actually sit. With the true `T=6..9` and `K=6` this same trade-off —
placing points to cover the smallest/hardest disagreement regions rather than
the numerous/easy ones — is exactly what the checker measures.
