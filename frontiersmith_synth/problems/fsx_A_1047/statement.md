# Scriptorium Atlas: Exemplars and Glosses

## Problem
A monastery scriptorium has produced `n` recensions of an illuminated manuscript,
numbered `1..n` in the order they were copied. Recension `1` is the original
**Urtext**. Every later recension `i` (`2 <= i <= n`) was copied from an earlier
recension `p_i < i`, its **source** — together the recensions form a tree rooted
at `1`.

Copying between a recension and its source can go two ways, at different labor
costs (interpolating new illuminations is not the same craft as excising
them). On the link between `i` and its source `p_i`:
- an **ascent gloss registered on `i`** lets a monk reconstruct `i` itself by
  starting from a full exemplar (or further reconstruction) of `p_i` and
  copying in the interpolations. This is exactly what happens when `i`'s own
  registered pointer targets `p_i`. It costs `up_i` labor-hours.
- a **descent gloss registered on `i`** lets a monk reconstruct `p_i` itself
  by starting from a full exemplar (or further reconstruction) of `i` and
  unwinding those same interpolations. This is exactly what happens when
  `p_i`'s own registered pointer targets `i` (the source routes through its
  own derivative). It costs `down_i` labor-hours.
`up_i` and `down_i` are independent positive integers given in the input — one
direction is often far cheaper than the other, and this is not tied to which
recension is "older."

Each year the monastery has a fixed allowance `S` of vellum and gold leaf. Any
subset of recensions may be kept as **full exemplars**; keeping recension `v`
as an exemplar costs `size_v` vellum. Every recension that is *not* an exemplar
must be assigned exactly **one** registered gloss pointing to a tree-adjacent
recension — either its source `p_i`, or one of its direct derivatives (some `j`
with `p_j = i`). Following registered glosses from any recension must terminate
at an exemplar (no cycles, no dead ends).

Pilgrims request recension `v` `w_v` times per year (this demand is heavily
skewed — a handful of recensions, not necessarily the deepest or the newest,
draw most requests). Reconstructing `v` for a pilgrim costs the sum of the
gloss labor-hours along `v`'s chain to the exemplar it reaches (`0` if `v`
itself is an exemplar).

## Input (stdin)
```
n S
size_1 w_1
p_2 up_2 down_2 size_2 w_2
p_3 up_3 down_3 size_3 w_3
...
p_n up_n down_n size_n w_n
```
`size_1 = 1` always (the Urtext is always affordable to keep alone), so `S >= 1`
guarantees at least one feasible construction.

## Output (stdout)
```
k
c_1 c_2 ... c_k
v_1 t_1
v_2 t_2
...
v_{n-k} t_{n-k}
```
`k` and the list of `k` exemplar ids, then one line per **non-exemplar**
recension `v` giving its gloss target `t` (must be tree-adjacent to `v`: either
`t = p_v` or `v = p_t`). Every recension must appear exactly once, either as an
exemplar or as a `(v, t)` pair.

## Feasibility
Reject (score `0`) on any of: malformed tokens; exemplar ids out of range or
duplicated; `sum(size_c for c in exemplars) > S`; fewer than one exemplar;
a non-exemplar recension missing, duplicated, or repeated as a pointer source;
a pointer target not tree-adjacent to its source; a gloss cycle (a chain that
never reaches an exemplar).

## Objective
Minimize `F = sum_v w_v * cost(v)`, where `cost(v)` is the sum of gloss
labor-hours on `v`'s chain to the exemplar it reaches (`cost(v) = 0` for
exemplars). Lower `F` is better.

## Scoring
The checker builds its own baseline `B`: keep only the Urtext as an exemplar,
and chain every other recension upward through ascent glosses to its source
(so `B = sum_v w_v * (ascent cost from v to the root)`). With minimization
normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing the baseline scores `Ratio = 0.1`; cutting the weighted labor to a
tenth of the baseline caps the score at `1.0`.

## Constraints
- `1 <= n <= 1800`, `1 <= S`, all costs/sizes positive integers, `0 <= w_v`.
- Time limit 5s, memory 512m.

## Example
`n=3`, Urtext `1` (`size_1=1, w_1=5`), recension `2` sourced from `1`
(`up_2=10, down_2=1, size_2=5, w_2=100`), recension `3` sourced from `2`
(`up_3=10, down_3=1, size_3=5, w_3=1`), budget `S=6`. Baseline: exemplar
`{1}`, both `2` and `3` ascend to their source: `cost(2)=up_2=10`,
`cost(3)=up_3+cost(2)=20`, `B = 100*10 + 1*20 = 1020` (`1` contributes `0`,
it is the exemplar). A smarter choice keeps `{2}` as the exemplar
(`size_2=5 <= 6`): `1` is not an exemplar and its only tree-neighbor is `2`,
so `1`'s pointer targets `2` — a descent gloss registered on `2`, costing
`down_2=1`; `3` still ascends to `2`, costing `up_3=10`. `F = w_1*1 + w_3*10
= 5*1 + 1*10 = 15` (`2` contributes `0`, it is now the exemplar).
`Ratio = min(1000, 100*1020/15)/1000 = 1.0` (capped).
