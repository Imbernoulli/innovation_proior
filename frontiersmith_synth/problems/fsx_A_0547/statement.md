# Manifold Inversion: Setting Inlets to Hit Leaf Targets

## Problem
A chemical plant mixes reagents through a **fixed manifold** modeled as a DAG.
Nodes `0..S-1` are **inlet tanks** (sources). Nodes `S..S+I-1` are internal
**mixers**; nodes `S+I..S+I+L-1` are **leaf outlets**. Every non-inlet node is a
**weighted average** of its parents: its flow-fractions are fixed, nonnegative,
and sum to `1`. So each species' concentration at a node is that convex
combination of its parents' concentrations.

You control the inlets. For each inlet `i` and species `s` you set a
concentration `x[i][s]` with `0 <= x[i][s] <= cap`. Concentrations then
propagate deterministically through the manifold, giving an achieved
concentration `ach[l][s]` at every leaf `l`. Each leaf has a **target**
`t[l][s]`. You want the leaves to match their targets while injecting little
total chemical.

Because many leaves share ancestors, the leaf outputs are **coupled**: the
achievable set `{ ach : 0 <= x <= cap }` is a bounded convex body (a zonotope),
not all of space. Some targets lie outside it and can only be approached.

## Input (stdin)
```
S I L M cost cap
```
Then `I+L` lines, one per node `j = S, S+1, ..., S+I+L-1` in order, each:
```
d  p_1 w_1  p_2 w_2  ...  p_d w_d
```
`d` parents (each index `< j`) with flow-fractions `w_k >= 0`, `sum_k w_k = 1`.
Then `L` lines; line `l` gives the `M` targets `t[l][0] ... t[l][M-1]` for leaf
node `S+I+l`.

## Output (stdout)
`S` lines; line `i` holds the `M` reals `x[i][0] ... x[i][M-1]`.

## Feasibility
Print exactly `S*M` reals. Each must be **finite** and satisfy
`0 <= x[i][s] <= cap` (tolerance `1e-6`). Any missing / non-finite / out-of-range
value scores `Ratio: 0.0`.

## Objective (minimize)
Propagate the inlets (`ach[node][s] = sum over parents w * ach[parent][s]`,
`ach[inlet][s] = x[inlet][s]`). Then
```
F  =  sum_{leaves l, species s} ( ach[l][s] - t[l][s] )^2
      +  cost * sum_{inlets i, species s} x[i][s]
```
The first term is the total squared leaf mismatch; the second penalizes total
chemical injected. Smaller `F` is better.

## Scoring
The checker's baseline is **inject nothing** (`x = 0`), giving
`B = sum_{l,s} t[l][s]^2`. With minimization normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing `x = 0` scores `Ratio = 0.1`; driving `F` to `B/10` caps at `1.0`.
Targets are engineered to be partly unreachable, so an irreducible residual
keeps the ceiling open.

## Constraints
- `5 <= S <= 22`, `8 <= I <= 96`, `L = S`, `2 <= M <= 6`, `cap = 1`.
- `cost = 0.02`. All weights nonnegative and sum to 1 per node.
- Time limit 5s, memory 512m. Each input < 5 MB.

## Note on strategy
The leaf-to-inlet map is linear, so the tempting move is an **unconstrained**
least-squares inverse per species followed by clipping into `[0,cap]`. Shared
ancestors make that inverse leave the box on the few high-influence inlets, and
clipping them corrupts every leaf they feed. A better solution treats it as a
**box-constrained projection** onto the achievable set and trades those
bottleneck inlets off against the many leaves downstream.

## Example
`S=1, I=0, L=1, M=1, cost=0, cap=1`. One inlet feeds one leaf directly
(leaf node's single parent is inlet 0 with weight 1). Target `t = 2.0`, so
`B = 4.0`. The best feasible inlet is `x = 1.0` (`ach = 1`), giving
`F = (1-2)^2 = 1`, `sc = 100*4/1 = 400`, `Ratio = 0.4`. Setting `x = 0`
reproduces the baseline `Ratio = 0.1`.
