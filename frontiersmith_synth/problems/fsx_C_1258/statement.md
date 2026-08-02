# Same-Set Bloom Cascade: Bits Where the Traffic Actually Goes

## Problem
A lookup service protects one fixed key set `S` (`n` keys) with a **cascade of
`L = 4` independent Bloom filters**, all guarding the *same* question "is `x`
in `S`?". A query walks the layers in order `1, 2, 3, 4`. Visiting layer `i`
costs `cost[i]` lookup units (`cost = [1, 3, 9, 27]` in every instance --
deeper layers are much pricier, since they live further down a storage
hierarchy). If layer `i` reports "definitely absent" the query stops there.
If it reports "maybe present" (all of that layer's hash bits are set) the
query pays for the visit and continues to layer `i+1`. A query actually in
`S` always reports "maybe present" everywhere (Bloom filters never
false-negative), so it always walks all 4 layers. A query **not** in `S` but
surviving all 4 layers anyway (a false positive at every layer) triggers an
expensive fallback lookup, adding a fixed penalty `D` to its cost.

You choose, for every layer `i`, a bit-array size `m_i` and a hash-function
count `k_i`, subject to a shared bit budget `B = sum(m_i)`. Layer `i`'s bit
`pos` is set for key `x`'s `j`-th hash (`0 <= j < k_i`) at
`pos = ((A[i][j]*x + B[i][j]) mod P) mod m_i`, using the fixed coefficients
given in the input and `P = 2^31 - 1`. Each layer is built once by inserting
every key of `S`.

Real traffic is **not** uniform: a small set of "hot" keys (given explicitly,
with their query weight) account for most of the load; everything else is
background "tail" traffic (weight 1). You are shown one sample of tail
traffic to calibrate against. Scoring replays the workload against a
**freshly drawn tail sample** from the identical distribution (the hot keys
and weights are unchanged, only the anonymous tail is redrawn) -- your
allocation must generalize to the true traffic pattern, not just match the
sample you saw.

## Input (stdin)
```
testId
n universe L kmax B
cost[1] cost[2] cost[3] cost[4]
D
key_1 key_2 ... key_n              (S, ascending, distinct, in [0, universe))
```
then, for each of the `L` layers in order, `kmax` lines `A B` (hash
coefficients; layer `i` uses only the first `k_i` of its `kmax` pairs), then
```
H
hotkey_1 weight_1
...
hotkey_H weight_H
T
tailkey_1 weight_1
...
tailkey_T weight_T
```
(the visible tail sample; `H` may be `0`).

## Output (stdout)
Exactly `L = 4` pairs of integers, in order, one layer's `(m_i, k_i)` each
(any whitespace layout):
```
m_1 k_1
m_2 k_2
m_3 k_3
m_4 k_4
```

## Feasibility
Exactly `2L` integer tokens. Each `m_i >= 8`; each `k_i` in `[1, kmax]`;
`sum(m_i) <= B`. Parse failure, wrong token count, non-integer/nan/inf
token, or violated bound all score `Ratio: 0.0`.

## Objective
Minimize `F`, the total weighted lookup cost (layer-visit costs plus any
`D`-penalties) over the held-out scoring sample.

## Scoring
The checker also builds its own reference: `B` split evenly over the 4
layers with `k=1` everywhere, scored the same way, giving `F_ref`. Then
```
Ratio = min(0.9, 0.1 * F_ref / F)
```
so the even/`k=1` split scores `0.1`, a tenth of its cost reaches `0.9`, and
`0.9` is a hard ceiling (no submission, however good, can reach `1.0`).

## Constraints
`1 <= n <= 1500`, `universe` up to `37500`, `L = 4` (fixed), `kmax = 6`,
`B` up to `30000`, `1 <= H <= 5` or `H = 0`, tail sample up to `1200` entries.
Deterministic integer arithmetic only; no timing.

## Example
Toy case, `n=1`, `S={5}`, one hash per layer, `cost=[1,3,9,27]`, `D=500`. A
query for `x=5` (a true member) always survives every layer: cost
`1+3+9+27 = 40`. A query for `x=9` that collides with `5`'s bit at layers
1-3 but misses at layer 4 costs `1+3+9 = 13` (rejected before paying layer
4). If `x=9` collided at all 4 layers it would cost `1+3+9+27+500 = 540`.
Shrinking layer 1 (cheapest) enough to reject most non-members immediately
beats an even split that lets them wander deep before being caught -- or
leak through -- but starving layers 2-4 to near-nothing makes any
survivor's leak near-certain, so the useful move is a shift toward layer 1,
not a collapse onto it.
