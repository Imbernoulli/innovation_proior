# One Plan, Many Shapes: Cheapest Adaptive Probe Tree for a Sparsity Sweep

## Problem
A block computation has **B** blocks. You ship it to a fleet that will run it under
many different **sparsity shapes**. Under a given shape, each block is either
**zero** (contributes nothing) or **nonzero** (its multiply must be performed).
You are given the whole published **sweep** of **P** shapes up front and must
compile **one** branching plan that is correct on every shape while minimizing the
worst-case op-cost.

The plan is a decision DAG over three ops:
- `T j a b` — **test** whether block `j` is zero in the running shape (**cost 1**);
  branch to node `a` if it is zero, else to node `b`.
- `M i c` — **multiply** block `i` (**cost M**); continue to node `c`.
- `H` — **halt**.

Execution starts at node 0; the running shape answers the tests. On a shape, the
cost is `(#tests executed) + M*(#multiplies executed)` along the path taken.

## Input (stdin)
```
B P M
<P lines, each a length-B string over {0,1}>   # '1' = block nonzero on that shape
```

## Output (stdout)
```
N
<N node lines, node i on line i+1, node 0 is the start>
```
Each node line is `T j a b`, `M i c`, or `H` (indices `0<=j,i<B`, `0<=a,b,c<N`,
`1<=N<=300000`). The DAG may be shared/reused but every path must terminate.

## Feasibility
For **every** one of the P shapes, the multiplies performed on that shape's path
must include **all** of its nonzero blocks (extra multiplies are allowed but
cost). Any malformed plan, out-of-range index, non-terminating path, or a shape
whose required block is left uncomputed scores **0**.

## Objective (minimize)
```
F = max over the P shapes of ( #tests + M * #multiplies ) on that shape's path
```

## Scoring
Let `U` be the set of blocks that are nonzero in at least one shape. The reference
baseline is the static plan "multiply all of `U`, run no tests", of cost
`B_ck = M*|U|`. Your score is
```
Ratio = min(1.0, 0.1 * B_ck / F)
```
So the static-union plan scores ≈ 0.1, and driving the worst case down 10× caps
the score at 1.0. Higher is better.

## Why this is not one multiplication to optimize
The multiplies a shape forces on you are fixed; the only levers are **which tests
you spend** and **whether you waste multiplies on blocks that happen to be zero on
the current shape**. Two naive plans bracket the difficulty:
- *Static union*: multiply everything in `U`, never test — pays full price on every
  shape (this is the ≈0.1 baseline).
- *Probe-all*: test all B blocks, multiply the ones that came back nonzero — correct,
  but every shape eats B tests, so the test bill dominates when B is large.

The shapes are **not** independent: across the sweep they differ only in a few
**discriminating** blocks, and the rest of each shape's nonzeros are determined
once its identity is pinned down. The real task is to find the **cheapest splitting
invariant** of the shape family — a shallow test order that identifies the shape in
a handful of probes — and then multiply exactly that shape's own blocks. Which
blocks discriminate is **not** marked; you must recover it from the P masks.

## Example
`B=4, P=2, M=14`, masks `1100` and `1011`. Here `|U|=4`, so the static baseline is
`B_ck = 56`. Block 1 discriminates the two shapes. A plan that first runs `T 1 …`
then, on the `1100` branch multiplies `{0,1}` and on the `1011` branch multiplies
`{0,2,3}`, has worst-case cost `1 + 14*3 = 43`, scoring `0.1*56/43 ≈ 0.130` —
better than probe-all (`4 + 14*3 = 46`) and better than the static union (`56`).

## Constraints
Time limit 2–5 s, memory ≤ 512 MB. Deterministic integer scoring. `B` up to a few
hundred; `P` up to a few dozen; each instance well under 1 MB.
