# Skeleton-Key Polynomials over F_p

## Problem
Fix a prime `p`. A "vault" has one door at every element of `F_p = {0,1,...,p-1}`.
A subset `S ⊆ F_p` (given to you) lists the **marked** doors — the ones a valid key
must open. Every unmarked door must stay shut.

You must forge a **skeleton key**: a polynomial `f(x)` over `F_p`, given as a
**sparse list of (exponent, coefficient) terms**, with **at most `T` terms**
(`T` is given in the input, `T << |S|`). A door `x` is *opened* by your key iff
`f(x) ≡ 0 (mod p)`. You want to open every marked door and no unmarked one.

## Input (stdin)
```
p T
m
x_1 x_2 ... x_m
```
`p` is prime, `T` is your term budget, `m = |S|`, and `x_1,...,x_m` (0-indexed,
`0 <= x_i < p`, all distinct) are exactly the marked doors `S`. `1000 <= p <=
30000`, `4 <= T <= 24`.

## Output (stdout)
```
k
e_1 a_1
e_2 a_2
...
e_k a_k
```
`0 <= k <= T`. Each `e_i` is an exponent with `0 <= e_i <= p-2`, each `a_i` a
coefficient with `0 <= a_i <= p-1`, and all `e_i` distinct. Your key is
`f(x) = sum_i a_i * x^{e_i} (mod p)`.

## Feasibility
Output must parse exactly (right token count, integers in range, no duplicate
exponents). Any violation scores `0`.

## Objective (maximize)
Let `correct = |{x in S : f(x) = 0}|` (marked doors opened) and
`wrong = |{x not in S : f(x) = 0}|` (unmarked doors opened). Your score is
```
F = correct - 2 * wrong          (F clamped to >= 0)
```
Opening an unmarked door costs twice as much as opening a marked one is worth —
a key that opens *everything* is worthless.

## Scoring
The judge builds its own trivial key `B`: it tries every binomial `x^d - c`
with `12 <= d <= 25` (root set of 12 to 25 elements), keeps the single best
one by the same `F` formula, and uses that value as `B` (at least `1`). Your
ratio is
```
ratio = min(1.0, 0.1 * F / B)
```
so matching the judge's narrow guess scores about `0.1`, and `10x` its `F`
value caps the ratio at `1.0`.

## What makes it hard
`S` is not scattered randomly. For a divisor `d` of `p-1`, the solutions of
`x^d = c` (when they exist) form **exactly one coset of the order-`d`
subgroup** of `F_p*` — a rigid algebraic set of size `d`, no more, no less.
The judge plants `S` in two steps. First it unions **2 to 4 cosets of
distinct subgroup orders** `U` (one deliberately small, 12-25; the others
bigger), then knocks about 3% of `U`'s own points back out (so even the
*exact* product key racks up a few unavoidable false positives on those
holes) and scatters in a handful more points outside `U` entirely, unrelated
to any subgroup. Multiplying in one more `x - z` factor to also root out a
single extra point roughly *doubles* your current term count (expanding a
product only ever grows multiplicatively), so once you're already spending
most of `T` composing cosets, chasing the scattered extras isn't affordable.
A single binomial can only ever capture *one* planted coset, however cleverly
you search for `d`. Since
roots of a **product** of binomials are exactly the **union** of their root
sets (over a field, no zero-divisors), multiplying together the binomials for
several well-fitting orders — then expanding to its (still short) monomial
list — opens most of `S` in well under `T` terms. Writing the polynomial that
vanishes on literally every point of `S` (plain interpolation) is exact but
needs about `|S|` terms, far over budget.

## Example
`p = 13`, `S = {2, 5, 6, 10}`. The cube-roots-of-unity subgroup of `F_13*` is
`{1,3,9}`; the coset `2*{1,3,9} = {2,5,6}` (each cubes to `8`). Submitting
```
2
3 1
0 5
```
means `f(x) = x^3 + 5 = x^3 - 8 (mod 13)`. Over all of `F_13`, `f(2)=f(5)=
f(6)=0 (mod 13)` and nothing else vanishes. So `correct = 3` (doors `2,5,6`),
`wrong = 0` — door `10` (noise) stays unreached by this 2-term key, since it
was never part of any coset. `F = 3`. Real instances plant several cosets at
once and are far larger than this toy.
