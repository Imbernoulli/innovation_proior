# Switching Looms: Dataflow Mapping for a Chained Matmul Sequence

## Problem
A systolic accelerator holds a fixed grid of `P` rows x `Q` columns of
multiply-accumulate cells. You must run `L` matrix multiplications back to
back; layer `i` multiplies an `M_i x K_i` matrix by a `K_i x N_i` matrix.
Before streaming data through the grid for a layer, two of its three
dimensions `{M_i, K_i, N_i}` are pinned **stationary** onto the grid's two
physical axes (the row axis holds `P` cells, the column axis `Q` cells); the
third dimension's values stream through the grid one step per cycle. If a
stationary dimension exceeds the axis it is pinned to, the grid must be
reloaded and reused across multiple **tiles** to cover it. If a stationary
dimension is smaller than its axis, the leftover cells still power up and
clock every cycle anyway — wired to zero, contributing nothing useful.

You choose, independently for each layer, a **mapping code**: a permutation
of the letters `M`,`K`,`N`, where the first letter names the dimension pinned
to the row axis, the second the dimension pinned to the column axis, and the
third is the streaming dimension. There are exactly 6 codes. Switching the
code between two consecutive layers requires draining and re-wiring the
grid's control/routing network — a fixed cost independent of matrix size.

## Input (stdin)
```
P Q L RELOAD SWITCH
M_1 K_1 N_1
...
M_L K_L N_L
```

## Output (stdout) — the artifact
```
L
code_1
...
code_L
```
Line 1 must repeat `L` (a lightweight self-check). Each of the next `L`
lines is one of `MKN, MNK, KMN, KNM, NMK, NKM` — the code used for that
layer, in order. Any other token, a wrong line count, a header that doesn't
equal `L`, or a code that is not a genuine permutation of the 3 letters makes
the artifact infeasible (score `0`).

## Objective (minimize)
For layer `i` with code `(d1,d2,s)` (a dims lookup `{M:M_i,K:K_i,N:N_i}`):
```
tp = ceil(dim[d1] / P)        tq = ceil(dim[d2] / Q)
per_tile = RELOAD + dim[s] + (P + Q - 1)
layer_cost_i = tp * tq * P * Q * per_tile
```
`tp*tq` counts how many tiles are needed to cover the whole stationary plane
(each tile pays `RELOAD` cycles of reload latency plus a `P+Q-1` pipeline
fill/drain); multiplying by `P*Q` charges every cell of the grid for every
cycle it is powered — including the idle, zero-padded cells on a partial
tile. Between two consecutive layers whose codes differ, add `SWITCH * P * Q`
once. The total to minimize is
```
F = sum(layer_cost_i for all i) + sum(SWITCH*P*Q for each code change)
```
computed with exact non-negative integers throughout.

The judge's own baseline `B` reuses this exact formula but forces the single
fixed code `KNM` (pin `K,N`; stream `M` — the textbook weight-stationary
default) on every layer regardless of shape. Score:
```
Ratio = min(1.0, 0.1 * B / max(1e-9, F))
```
Reproducing the baseline scores `0.1`; a tenth of its cost caps at `1.0`.
Chasing the single dataflow that best serves your biggest layer is not free
either: forcing that wiring on every smaller, differently-shaped layer in the
sequence pays the idle-cell tax repeatedly on layers it was never suited
for — sometimes costing more in aggregate than the reconfiguration price of
re-wiring for them individually would have.

## Constraints
`4<=P,Q<=24`, `3<=L<=16`, `2<=M_i,K_i,N_i<=400`, `1<=RELOAD<=10`,
`1<=SWITCH<=30`.

## Example (worked score, illustrative shape only)
`P=Q=4, RELOAD=2, SWITCH=3`, two layers `(M,K,N)=(4,4,4)` then `(4,4,16)`.
Using code `KNM` on both: layer 1 has `tp=tq=1`, `per_tile=2+4+7=13`,
`cost=1*16*13=208`. Layer 2 pins `K=4` (`tp=1`), `N=16` (`tq=4`), streams
`M=4`: `per_tile=2+4+7=13`, `cost=1*4*16*13=832`. No switch, so
`F=208+832=1040`; the baseline makes the same choice here, so `B=1040` and
`Ratio=0.1`. Switching layer 2 to code `MKN` instead (pin `M=4,K=4`; stream
`N=16`): `tp=tq=1`, `per_tile=2+16+7=25`, `cost=16*25=400`, plus one switch
`3*16=48`; layer 1 is unchanged at `208`. `F=208+400+48=656`, so
`Ratio=min(1, 0.1*1040/656)=0.1585` — a real improvement even after paying
to re-wire. Real graded cases mix many more, more lopsided layer shapes.
