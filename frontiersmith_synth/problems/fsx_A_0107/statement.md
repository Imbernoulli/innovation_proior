# Ridge-Line Watchtower Codes

## Problem
A national forest runs a fire-detection network along a single straight ridge. The
ridge is marked with integer mile-posts `0, 1, ..., M`. You must place **exactly `n`
watchtowers**, each on a distinct mile-post, forming a placement set

```
A = { a_1, a_2, ..., a_n } ,   a_i integer,   0 <= a_i <= M,   all distinct.
```

Two facts about the radio protocol drive the design:

- **Convergence codes (localization).** When a fire is spotted, any two towers at
  posts `a` and `b` transmit the *sum* `a + b` as a cross-bearing "convergence
  code". The set of achievable convergence codes is the **sumset**
  `A + A = { a + b : a, b in A }`. More *distinct* convergence codes means the
  network can resolve fire positions more finely, so more distinct sums is better.

- **Parallax codes (spectrum load).** The same pair also emits the *difference*
  `a - b` as a "parallax code". The set of parallax codes is the **difference set**
  `A - A = { a - b : a, b in A }`. Every distinct parallax code permanently reserves
  one shared radio channel, so more distinct differences is worse.

You want a placement that is rich in distinct convergence codes yet frugal in
distinct parallax codes.

## Input (stdin)
A single line with two integers:
```
n M
```
`n` = number of towers to place (you must use all `n`); `M` = largest mile-post.

## Output (stdout)
The `n` distinct tower mile-posts, as integers separated by whitespace (spaces
and/or newlines). Order does not matter.

## Feasibility
Your output is accepted only if it lists **exactly `n` integers**, **all distinct**,
and **each in `[0, M]`**. Any violation scores `0`.

## Objective (maximize)
Maximize the code-efficiency ratio
```
F(A) = |A + A| / |A - A|
```
where `|A + A|` and `|A - A|` are the exact cardinalities of the sumset and the
difference set. Because `0` is always a difference and differences come in
`+/-` pairs, "spread-out" placements tend to have `|A - A| > |A + A|` (ratio below
1). Beating a ratio of `1` requires deliberately structured (more-sums-than-
differences) placements.

## Scoring
Let `B` be the ratio of a fixed reference placement of `n` towers that the grader
builds itself (a Sidon / perfect-ruler layout, whose ratio is well below `1`). Your
score is
```
sc    = min(1000, 100 * F(A) / B)
Ratio = sc / 1000
```
Reproducing a reference-quality placement scores about `0.1`; a placement ten times
more code-efficient than the reference caps at `1.0`. Infeasible output scores `0`.

## Constraints
- `64 <= n <= 200`, `M = 4 * n * n`.
- Time limit 5 s, memory 512 MB.
- Scoring is exact integer arithmetic and fully deterministic.

## Example (worked score)
Suppose `n = 4`, `M = 64`, and the grader's reference ratio is `B = 0.75`.
- Placement `A = {0,1,2,3}` (an arithmetic run): `A+A = {0..6}` has `7` elements,
  `A-A = {-3..3}` has `7` elements, so `F = 7/7 = 1.0`. Score
  `sc = 100 * 1.0 / 0.75 = 133.3`, `Ratio = 0.133`.
- Placement `A = {0,2,3,4}`: `A+A = {0,2,3,4,5,6,7,8} = 8` elements,
  `A-A = {-4,-3,-2,-1,0,1,2,3,4} = 9` elements, `F = 8/9 = 0.889`, a *worse* ratio.

The interesting placements are the rare ones with `F > 1`.
