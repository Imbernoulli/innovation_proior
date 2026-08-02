# Spectral Tuning: Placing a Network's Ringing Frequencies

## Problem

A network of `n` masses (vertices) is connected by `m` springs (edges). Spring `e`
between vertices `u` and `v` has a positive stiffness `w_e`, constrained to a given
range `[lo_e, hi_e]`. The network's small vibrations are governed by its weighted
**graph Laplacian** `L`, an `n x n` symmetric matrix built by, for every edge `(u,v,w)`:
adding `w` to `L[u][u]` and `L[v][v]`, and subtracting `w` from `L[u][v]` and `L[v][u]`.
The eigenvalues of `L`, sorted ascending `0 = lambda_1 <= lambda_2 <= ... <= lambda_n`,
are the network's squared resonant frequencies (the smallest is always exactly 0,
because the all-equal-displacement mode costs no energy).

You are given the topology, the per-edge stiffness bounds, and a **target spectrum**
`tau_1=0 <= tau_2 <= ... <= tau_n`. Choose stiffnesses inside their bounds so the
network's actual spectrum lands as close to the target as possible.

**The coupling to be aware of:** `L = sum_e w_e * b_e b_e^T`, where `b_e` is the
edge's +-1 incidence vector, so every single `b_e b_e^T` term is positive
semidefinite. Raising any one `w_e` is a positive rank-1 bump to `L`, and by
eigenvalue interlacing that bump moves *every* eigenvalue, not just a chosen one,
and constrains how their relative order can change. Treating each target
frequency as an independent knob to turn ignores this and can fight itself.

## Input (stdin)

```
n m
u_1 v_1 lo_1 hi_1
...
u_m v_m lo_m hi_m
tau_1 tau_2 ... tau_n
```
`1 <= u_i,v_i <= n` (1-indexed, the graph is connected), `1 <= lo_i < hi_i`.
`tau_1` is always `0.000000`; `tau_2..tau_n` are sorted ascending non-negative reals.

## Output (stdout)

`m` space-separated real numbers `w_1 ... w_m`, the chosen stiffness for edge `i`
(in the same order the edges were listed), each satisfying `lo_i <= w_i <= hi_i`.

## Feasibility

Exactly `m` finite numeric tokens must be printed, each within its edge's bounds
(1e-6 tolerance). Any violation (wrong count, non-numeric, non-finite, or out of
range) scores 0.

## Objective

Build `L` from your weights, compute its exact eigenvalues, sort them ascending.
Let `err` be the root-mean-square difference between your `lambda_2..lambda_n` and
the targets `tau_2..tau_n` (the structural `lambda_1=0` carries no placement
information and is not scored). Your quality is `F = 1 / (1 + 4*err/tau_n)`
(higher is better; `tau_n` is the target's top frequency, used to normalize scale).

## Scoring

The checker also computes `B`, the same quality `F` achieved by the trivial
construction "every stiffness at the midpoint of its bounds, ignoring the target".
Your score is `min(1000, 100*F/B) / 1000`, i.e. matching the trivial midpoint
spectrum scores about 0.1, and getting the RMS error to zero saturates near 1.0.

## Constraints

`4 <= n <= 9`, `n-1 <= m <= n+1`, `1 <= lo_i <= 3`, `lo_i < hi_i <= lo_i+9`.
Time limit 5s, memory 512MB.

## Example

`n=3, m=2`: edges `(1,2,lo=1,hi=5)`, `(2,3,lo=1,hi=5)`, target `0 2 6`
(illustrative shape only, not from the real generator). Choosing `w=(2,4)` gives
`L = [[2,-2,0],[-2,6,-4],[0,-4,4]]` with eigenvalues `0, 2.536, 9.464` approximately
-- close to, but not exactly, the target, since the achievable spectra of a
2-edge path form only a 2-parameter family that cannot hit every 2-value target.
