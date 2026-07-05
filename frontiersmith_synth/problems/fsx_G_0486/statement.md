# Conflict-Free Ternary Tags (Cap Set in F_3^n)

## Problem
Your system labels items with **ternary tags**: strings of length `n` over the alphabet
`{0, 1, 2}`. Equivalently, each tag is a vector in `F_3^n` (the integers mod 3, one digit
per coordinate).

Three *distinct* tags `a`, `b`, `c` are said to **collide** when they are collinear in
`F_3^n`, i.e.

```
(a[k] + b[k] + c[k]) mod 3 == 0   for every coordinate k = 0 .. n-1.
```

(Over `F_3` this is the same as saying `a`, `b`, `c` form a 3-term arithmetic progression /
a full line.) A set of tags is **conflict-free** when it contains **no** colliding triple.
Such a set is classically called a *cap set*.

Your job: output as **large** a conflict-free set of tags as you can.

## Input (stdin)
A single integer `n` (the tag length / ambient dimension), possibly preceded by a comment
line beginning with `#`. Read the first integer token as `n`.

## Output (stdout)
```
m
tag_1
tag_2
...
tag_m
```
The first line is the count `m`. Each of the next `m` lines is a tag: a string of exactly
`n` characters, each in `{0, 1, 2}`.

## Feasibility
The output is accepted only if ALL hold (otherwise the score is 0):
- `1 <= m <= 3^n` and exactly `m` tags follow;
- every tag has length `n` and uses only characters `0`, `1`, `2`;
- all `m` tags are distinct;
- **no** three distinct tags collide (are collinear) as defined above.

## Objective
**Maximize** `F = m`, the number of tags (the cap size).

## Scoring
Let `B = 2^n`, the size of the `{0,1}^n` grid, which is always a valid conflict-free set
(a collision among 0/1 entries would force `a = b = c`). With `F` your validated cap size,

```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```

So the `{0,1}^n` baseline scores `0.1`, and a cap ten times denser than the grid would
saturate at `1.0`. For the `n` used here the true maximum cap size is **unknown**, and it is
far below that saturation point — there is real headroom, and many construction strategies
(algebraic products, tilings, local search) trade off against each other.

## Constraints
- `9 <= n <= 12` across the test ladder.
- Scoring is fully deterministic and exact (integer arithmetic over `F_3`).

## Example (worked score)
Suppose `n = 4` and you output the 20-element cap
```
20
0000
0001
...
2212
```
(the optimal cap of `F_3^4`). Then `F = 20`, `B = 2^4 = 16`, so
`sc = 100 * 20 / 16 = 125` and `Ratio = 0.125`. The `{0,1}^4` grid (`F = 16`) would instead
give `Ratio = 0.100`. This example is illustrative of the FORMAT and scoring only; the graded
instances use larger `n`.
