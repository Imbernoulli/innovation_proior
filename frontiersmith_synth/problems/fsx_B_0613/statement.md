# Pre-Ordered Wires: Sorting Networks That Skip Known Relations

## Problem

A **comparator network** on `n` wires is a list of comparators. A comparator `(i, j)`
with `i < j` reads the values on wires `i` and `j`, writes the **smaller** onto wire `i`
and the **larger** onto wire `j`. Executed in list order, the network *sorts* an input if
the final values are non-decreasing from wire `0` to wire `n-1`.

Here the inputs are not arbitrary: the wires arrive **pre-ordered**. You are given a
partial order — a set of relations `a ≺ b` meaning *"on every valid input, the value on
wire `a` is ≤ the value on wire `b`"*. Your network must sort **every** input consistent
with these relations (inputs where all the promised inequalities hold). Inputs that
violate a relation cannot occur, so your network is free to do anything on them.

Design the cheapest comparator network that provably sorts all consistent inputs.

## Input (stdin)

```
n E alpha
a_1 b_1
...
a_E b_E
```

- `n` (`6 ≤ n ≤ 24`) wires, indexed `0 … n-1`.
- `E` relations. Each line `a b` (with `0 ≤ a, b < n`) promises `value(a) ≤ value(b)` on
  every valid input. The partial order is the transitive closure of these relations.
- `alpha` (a positive real, given per instance) is the **depth weight** in the objective.

## Output (stdout)

```
m
i_1 j_1
...
i_m j_m
```

`m` = number of comparators, then one comparator per line with `0 ≤ i < j < n`.
An empty network is `m = 0` with no further lines.

## Feasibility

Let a **consistent** input be any assignment respecting all relations. Your network is
**feasible** iff it sorts every consistent input. By the 0-1 principle (which still holds
under a partial order, because thresholding is monotone), this is equivalent to sorting
every **monotone 0-1 input** — every 0-1 vector `x` with `x[a] ≤ x[b]` whenever `a ≺ b`.
The checker verifies this exactly over all such 0-1 inputs. Any infeasible, malformed,
out-of-range, non-integer, or `nan`/`inf` output scores `0`.

## Objective (minimize)

```
cost F = m + alpha * depth
```

- `m` is the comparator count.
- `depth` is the **parallel depth**: process comparators in the given order; comparator
  `(i, j)` sits in layer `1 + max(layer_ready[i], layer_ready[j])`; `depth` is the largest
  layer used. Fewer comparators is the primary win; `alpha` prices a secondary size-vs-depth
  trade-off, so the *order* you list comparators in also matters.

## Scoring

Let `B` be the cost of the full **insertion-sort network** on `n` wires
(`n(n-1)/2` comparators, depth `2n-3`) under the same `alpha` — a valid network that
ignores the partial order. With your feasible cost `F`:

```
Ratio = min(1.0, 0.1 * B / F)
```

The insertion baseline scores `0.1`; a 10x-cheaper network caps at `1.0`. The final grade
is the mean `Ratio` over 10 instances.

## The trap

Dropping in a general-purpose sorting network (insertion sort, or Batcher's odd-even
mergesort) sorts everything — but it wastes comparators **re-verifying relations you were
already promised**. Every comparator that only ever compares two wires whose order is
already fixed is dead weight. A sorting network needs to resolve exactly the pairs the
partial order leaves *incomparable*; known relations are comparators you can delete. The
instances plant already-sorted runs of varying length, so the gap between "run a textbook
network" and "only resolve what's unknown" is large — and `alpha` decides how much of the
saved budget you reinvest in shrinking depth.

## Example

Input `n=4`, relation `0 ≺ 1` (wires 0,1 already ordered), `alpha=0.10`.
Consistent monotone 0-1 inputs are those with `x[0] ≤ x[1]`. A feasible network:
```
4
2 3
0 2
1 3
1 2
```
Comparators = 4, depth = 3, `F = 4 + 0.10*3 = 4.3`. Baseline `B = 6 + 0.10*5 = 6.5`,
so `Ratio = min(1, 0.1*6.5/4.3) = 0.151`. (A full unconstrained sort of 4 wires needs 5
comparators; the known relation `0 ≺ 1` lets one be dropped.) Illustrative — not an instance.
