# Contraction-Order Optimization for a Quantum Tensor Network

## Problem
Simulating a quantum circuit reduces to contracting a **tensor network**: a set
of tensors joined by shared indices (bonds). The number of scalar
multiplications needed to evaluate the network depends *dramatically* on the
**order** in which you contract the tensors pairwise. Your job is to output a
contraction schedule that minimizes the exact total scalar-multiplication count.

The network has `m` tensors (ids `0..m-1`) and `k` indices. Each index `i` has a
dimension `d_i`. An index that lies on exactly two tensors is an internal
**bond** (it is summed away when those two tensors are contracted and no other
live tensor still carries it); an index that lies on exactly one tensor is an
open **leg** and is always kept.

## Input (stdin)
```
m k
d_0 d_1 ... d_{k-1}
deg_0 i_{0,1} ... i_{0,deg_0}
...
deg_{m-1} i_{m-1,1} ... i_{m-1,deg_{m-1}}
```
Line `2+u` lists the `deg_u` index ids carried by tensor `u`.

## Output (stdout)
Exactly `m-1` lines, each `a b`: contract the two currently-live tensors with
ids `a` and `b`. Initial tensors keep ids `0..m-1`; the result of the `t`-th
output line (0-indexed) is a **new** live tensor with id `m+t`. Every leaf must
be consumed and the schedule must end with a single remaining tensor.

## Feasibility
A schedule is valid iff it has exactly `m-1` lines, each references two
*distinct* currently-live ids, and exactly one tensor remains at the end.
Non-integer / out-of-range / `nan` / `inf` tokens are rejected.

## Objective (minimize)
When two tensors with live index sets `A` and `B` are contracted, the cost is
`prod_{i in A∪B} d_i` scalar multiplications. The total cost is the sum over all
`m-1` contractions. **Lower is better.**

## Scoring
Let `F` be your total cost and `B` the cost of the naive sequential schedule
`(0,1),(m,2),(m+1,3),...` that the checker recomputes itself. The reported
score is
```
Ratio = min(1, 0.1 * B / F)
```
So reproducing the sequential baseline scores ~0.1, and a schedule 10x cheaper
than the baseline caps the score at 1.0. The true optimum of a general
(cyclic) network is NP-hard and unknown, so there is genuine headroom.

## Constraints
`8 <= m <= 22`, `2 <= d_i <= 4`. All arithmetic is exact big-integer; scoring is
deterministic.

## Example (worked score)
For a tiny network with `B = 2048` and a submitted schedule of cost `F = 512`,
`Ratio = min(1, 0.1 * 2048/512) = min(1, 0.4) = 0.4`.
