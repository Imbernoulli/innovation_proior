# Datum Tree Error Budget: What to Measure Everything From

## Problem
A machinist must cut `n` features on a part. Feature `0` is the fixed **master
datum**. Every other feature is *located from* exactly one previously-usable
feature — its **datum**. These datum relations form a rooted tree: feature `i`'s
datum `par[i]` is one of its **allowed datums** (a listed set, all with smaller
index than `i`, encoding precedence — a feature must exist before it can serve as
a reference).

Cutting feature `i` introduces a worst-case positioning error along that edge.
On the ordinary machine the error is `a[i]`; a **precise machine** cuts it to
`p[i] < a[i]`. You own only `k` precise-machine slots to hand out across the
`n-1` cutting ops.

For a pair of features, their **relative error** is the accumulated worst-case
error along the unique datum-tree path between them (every op on the path
contributes its edge error). You are given `C` **critical pairs** `(u, v, w)`; a
pair's cost is `w` times its relative error. Choose the datum tree and the
precise-slot assignment to minimize the **weighted maximum** cost over all
critical pairs.

## Input (stdin)
```
n k
a[0] a[1] ... a[n-1]
p[0] p[1] ... p[n-1]
(for each feature i = 0..n-1)  d_i  g_1 ... g_{d_i}   # its d_i allowed datums
C
(for each critical pair)       u v w
```
Feature `0` has `d_0 = 0` (it is the root). For `i >= 1`, `d_i >= 1` and every
listed allowed datum is a valid index `< i`.

## Output (stdout)
```
par[0] par[1] ... par[n-1]
q  f_1 f_2 ... f_q
```
- `par[0]` must be `-1`. For `i >= 1`, `par[i]` must be one of feature `i`'s
  allowed datums. (Because allowed datums have smaller index, this is always a
  rooted tree.)
- The second line lists the `q <= k` features whose cutting op uses the precise
  machine (`q` first, then the distinct feature indices, none of them `0`). Use
  `0` for none.

## Feasibility
Any violation — wrong token count, `par[0] != -1`, an illegal datum, `q > k`, a
repeated or root precise slot, a non-integer token — scores `0`.

## Objective (minimize)
Let `eff[i] = p[i]` if `i` is precise, else `a[i]`. For pair `(u, v, w)` let
`S(u,v)` be the sum of `eff[x]` over every feature `x` on the tree path between
`u` and `v` except their lowest common ancestor. The score is
`F = max over pairs of w * S(u,v)`.

## Scoring
The checker builds a baseline `B` (each feature attached to its first-listed
allowed datum, no precise slots) and reports
`Ratio = min(1, 0.1 * B / F)`.
So the baseline scores about `0.1` and driving the weighted worst case `10x`
below it caps at `1.0`.

## Constraints
`n <= 4000`, `C <= 400`, time limit 5s, memory 512MB, each input under 5MB.

## Example
Two features share a long common trunk of datum edges before branching to their
own tips. The obvious moves — attach each tip as close to the master datum as its
precedence allows, then buy precise machining for the longest, most error-prone
chains — leave the shared trunk on the ordinary machine, so every critical pair
still pays the full trunk stack-up. Spending the same slots on the trunk edges
that lie on the most weighted pair-paths lowers the weighted maximum for all of
them at once. (Illustrative only — the profitable trade-off is set by the input.)
