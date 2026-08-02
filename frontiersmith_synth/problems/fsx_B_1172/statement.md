# Reading Rock Layers off a Bounce-Time Table

## Problem

The ground beneath a survey line is a stack of `N` horizontal layers. Layers
`1..N-1` are finite, with thickness `h_i > 0` and seismic velocity `v_i > 0`;
layer `N` is an infinitely deep half-space with velocity `v_N`. You must
reconstruct this earth model from two kinds of noiseless field measurements
recorded at a source point.

**Refraction first-arrivals.** At surface offset `x`, the first energy to
arrive is either the *direct wave* along the top layer (`t = x / v_1`), or a
*head wave* critically refracted along the top of some deeper layer `k`
(`1 < k <= N`) and returning to the surface. A head wave can only exist along
the top of layer `k` if `v_k` exceeds the velocity of **every** layer above
it (a real critical angle is needed at each shallower interface). If that
fails anywhere -- e.g. a low-velocity zone where a layer is slower than the
one directly above it -- no head wave forms there, and that interface leaves
no trace in the refraction data. The travel time of the head wave riding the
top of layer `k` (using layers `1..k-1` above it) is

```
t_k(x) = x / v_k + 2 * sum_{j=1}^{k-1} h_j * sqrt(v_k^2 - v_j^2) / (v_j * v_k)
```

and the first-arrival curve is the pointwise minimum of `x/v_1` and every
`t_k(x)` for which layer `k` satisfies the velocity condition above.

**Reflection two-way times.** For every interface `k = 1..N-1` (regardless of
any low-velocity zone -- ordinary reflection needs no critical angle) you are
also given `tau_k = 2 * sum_{j=1}^{k} h_j / v_j`. Notice `tau_k` only fixes
the *ratio* `h_j / v_j` per overlying layer: a layer twice as thick and
twice as fast gives an identical delay. Reflection data alone can never
separate `h_j` from `v_j` -- only the refraction slopes, where they exist,
break that tie.

*Illustrative worked FORM only (not the hidden law, not a real test case):* a
single layer over a half-space, `h_1 = 100`, `v_1 = 1000`, `v_2 = 2000`, gives
`tau_1 = 2*100/1000 = 0.2` and, at `x=1000`, `t_2(1000) = 1000/2000 +
2*100*sqrt(2000^2-1000^2)/(1000*2000) = 0.5 + 0.173 = 0.673`.

## Input (stdin)

```
test_id
N
M
x_1 t_1
x_2 t_2
...
x_M t_M
tau_1 tau_2 ... tau_{N-1}
V_MIN V_MAX H_MIN H_MAX
```
`(x_i, t_i)` are `M` observed refraction first-arrival picks, sorted by
increasing offset. Every true `v_i` lies in `[V_MIN, V_MAX]` and every true
`h_i` lies in `[H_MIN, H_MAX]`.

## Output (stdout)

Exactly `N` lines: for `i = 1..N-1`, one line `h_i v_i` (the finite layers,
top to bottom); then one final line with a single number `v_N` (the
half-space velocity).

## Feasibility

All `N-1` thicknesses and all `N` velocities must be finite positive numbers
within a generous sanity range (well beyond `[H_MIN,H_MAX]`/`[V_MIN,V_MAX]`,
to allow honest uncertainty about any one hidden layer). Anything else --
wrong token count, non-numeric, non-finite, non-positive, or out of range --
scores `0`.

## Objective and Scoring

Your score rewards two things against the (hidden) true model: how well your
cumulative depths `D_k = h_1 + ... + h_k` match the true interface depths,
and how well the first-arrival curve your model predicts matches the true
one at offsets you were *not* shown. Both components decay smoothly (no hard
cutoff) as your reconstruction drifts from the truth, and are rescaled
against an internal reference so a naive constant-velocity guess scores low
and a faithful reconstruction scores much higher, with room left above any
reference solution. The exact decay rate and reference construction are not
disclosed; the shape of the score -- depth fidelity plus held-out
travel-time fidelity, both smooth and graded -- is everything you need.

## Constraints

`3 <= N <= 6`, `20 <= M <= 40`, coordinates/times given to 6 decimals,
`800 <= V_MIN < V_MAX <= 6000`, `20 <= H_MIN < H_MAX <= 400`. Time limit 5s,
memory 512MB.

## Example (worked score, qualitative)

Exactly reproducing every true depth and velocity pushes both score
components near their maximum, well above any reference solution but never
capped at the very top -- headroom is intentional. Guessing one constant
velocity for the whole subsurface fits near-offset times reasonably but
drifts badly at depth, scoring only modestly above the internal baseline.
