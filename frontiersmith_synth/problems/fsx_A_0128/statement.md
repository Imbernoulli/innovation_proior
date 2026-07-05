# Tide-Pool Interaction Channels: Minimal-Rank Tensor Decomposition

## Problem
A stretch of coastline is modelled as a set of interacting tide pools. Field
biologists have measured a three-way interaction tensor `T` of shape
`a x b x c`, where `T[i][j][k]` records the net coupling strength between
**species** `i`, **nutrient** `j`, and **tidal state** `k`.

Ecologists believe the whole tensor is driven by a small number of hidden
*interaction channels*. A single channel is a rank-1 pattern: a species profile
`u` (length `a`), a nutrient profile `v` (length `b`) and a tidal profile `w`
(length `c`), contributing `u[i] * v[j] * w[k]` to every entry.

Your task: reconstruct `T` **exactly** as a sum of as **few** channels as
possible. The number of channels is the number of scalar multiplications a
mechanistic model would need — fewer is a more parsimonious explanation.

## Input (stdin)
```
a b c
```
followed by the tensor entries. For each tidal state `k = 0..c-1`, there are `a`
lines of `b` integers giving `T[i][j][k]` (species `i` down the rows, nutrient
`j` across the columns). All entries are integers. `1 <= a,b,c <= 5`.

## Output (stdout)
```
R
<channel 1>
<channel 2>
...
<channel R>
```
`R` is the number of channels. Each channel line has exactly `a + b + c`
rationals: first the species profile `u[0..a-1]`, then the nutrient profile
`v[0..b-1]`, then the tidal profile `w[0..c-1]`. Each number may be an integer
or a fraction written `p/q` (e.g. `-3`, `1/2`, `-7/4`).

## Feasibility
Let the reconstruction be
`Trec[i][j][k] = sum over channels r of u_r[i] * v_r[j] * w_r[k]`.
The output is feasible only if `Trec[i][j][k] == T[i][j][k]` for **every**
`(i,j,k)`, evaluated with exact rational arithmetic. `R` must be at least 1.
Any parse error, wrong line width, wrong channel count, or reconstruction
mismatch scores `0`.

## Objective
Minimize `R`, the number of interaction channels.

## Scoring
Let `B` be the number of nonzero entries of `T` (the naive construction that
uses one channel per nonzero entry). Your feasible submission with `R` channels
scores
```
ratio = min(1.0, 0.1 * B / R)
```
So the naive one-channel-per-entry solution scores about `0.1`, and any
decomposition using `<= B/10` channels caps the ratio at `1.0`. Lower `R` is
strictly better.

## Constraints
- `1 <= a, b, c <= 5`.
- Deterministic exact-rational scoring; no randomness, no timing.

## Example
Suppose `a=b=c=2` and `T` is the rank-2 tensor
`T = 1_a (x) 1_b (x) 1_c + e0 (x) e0 (x) e0` (all-ones plus a bump at the
corner). Then `B = 4` nonzeros can be covered by `R = 2` channels
(`u=v=w=[1,1]` and `u=v=w=[1,0]`), scoring
`ratio = min(1, 0.1 * 4 / 2) = 0.2`, versus `0.1` for the 4-channel naive
solution.
