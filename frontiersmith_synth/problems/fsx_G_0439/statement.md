# Shared Addition Sequence for Batch Modular Exponentiation

## Problem
In modular exponentiation you compute `x^e mod m` by a sequence of squarings and
multiplications. When a single base `x` must be raised to **several** exponents
`T = {t_1, ..., t_k}` at once (batch exponentiation, multi-signature verification,
window tables), all the target powers can share intermediate products.

Formally, an **addition sequence** for the target set `T` is a sequence of positive
integers

```
a_0 = 1, a_1, a_2, ..., a_L
```

such that every element after the first is the sum of two (not necessarily distinct)
**earlier** elements — i.e. for each `i >= 1` there exist indices `j, k < i` with
`a_i = a_j + a_k` — and every target `t in T` appears somewhere in the sequence.

Each such addition corresponds to one modular multiplication when evaluating the powers
`x^{a_i}`. The **cost** is the number of additions `L` (the count of elements after the
initial `1`). Your job is to make `L` small.

Computing a shortest addition sequence is NP-hard, so no formula gives the optimum; strong
solutions come from clever sharing of partial sums across the targets.

## Input (stdin)
```
k
t_1 t_2 ... t_k
```
`k` distinct integers with `2 <= t_i <= 4000`.

## Output (stdout)
The whole addition sequence as whitespace-separated integers, starting with `1`:
```
1 a_1 a_2 ... a_L
```
Order matters: the "earlier" constraint is evaluated in the order you print them.

## Feasibility
The sequence is **valid** iff:
1. the first element is `1`;
2. every later element equals `a_j + a_k` for some two earlier elements (`j, k` strictly
   before it; `j = k` is allowed, i.e. doubling);
3. every element satisfies `1 <= a_i <= max(T)`;
4. every target `t in T` occurs in the sequence.

Any violation, a non-integer / non-finite token, an empty sequence, or an absurdly long
sequence scores `0`.

## Objective
**Minimize** the cost `L = (number of elements) - 1`.

## Scoring
Let `B` be the length of the checker's baseline construction: an **independent binary**
addition chain for each target concatenated together (no sharing). With `F` your cost,
```
score = min(1.0, 0.1 * B / F)
```
Reproducing the baseline gives `0.1`; you must be `10x` shorter than the naive baseline to
saturate at `1.0` (unreachable here), so real headroom always remains.

## Constraints
`k` up to ~44, targets up to `4000`, sequence length `<= 20000`.

## Example
Targets `T = {7, 10}`.
- Baseline (independent binary): for `7 = 111b`: `1,2,3,6,7` (4 adds); for
  `10 = 1010b`: `1,2,4,5,10` (4 adds). Concatenated `B = 8`.
- A shared sequence `1 2 4 5 7 10` reaches `5=1+4`, `7=... `. For instance
  `1 2 3 4 7 5 10`: `2=1+1, 3=1+2, 4=2+2, 7=3+4, 5=1+4, 10=5+5` — cost `L = 6`,
  score `= min(1, 0.1 * 8 / 6) = 0.133`.

(The example is illustrative; targets in the tests are larger and more numerous.)
