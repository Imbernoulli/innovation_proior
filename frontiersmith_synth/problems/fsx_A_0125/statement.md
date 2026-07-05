# Wind-Tunnel Sensor Leakage Energy

## Problem

A wind-tunnel test section carries a linear array of `n` pressure sensors,
indexed `0 .. n-1`. You choose a **gain profile**
`f = (f_0, f_1, ..., f_{n-1})` with each `f_i >= 0`: the amplification you apply
to the signal from sensor `i`.

When the array demodulates a passing pressure wave, two sensors at positions `i`
and `j` beat together and deposit spurious ("aliased") energy into the combined
band `k = i + j`. The **leakage amplitude** deposited into band `k` is the
autoconvolution of the gain profile with itself:

```
c_k = sum over all i,j with i+j=k of  f_i * f_j        (k = 0 .. 2n-2)
```

Unlike a single worst hotspot, the calibration standard here charges you for the
**total leakage energy summed over every band** — the squared `L2` norm of the
leakage spectrum. Concentrated gains dump energy into a few bands and are
penalized quadratically. Define the **normalized leakage-energy index**

```
E(f) = n * ( sum_k c_k^2 )  /  ( sum_i f_i )^4
```

The index is scale-invariant (multiplying every `f_i` by a common constant does
not change it), so only the *shape* of the gain profile matters.
**Minimize `E(f)`.**

This is a step-function instance of the second autocorrelation / additive-energy
inequality. The uniform profile is a strong, obvious choice, but it is **not**
optimal: a carefully shaped, non-obvious profile spreads the leakage spectrum
more evenly and lowers the energy further. There is a hard positive floor — no
profile drives `E` to zero.

## Input (stdin)

A single integer `n` (`12 <= n <= 30`): the number of sensors.

## Output (stdout)

`n` real numbers separated by whitespace: the gains `f_0 .. f_{n-1}`.

## Feasibility

- Exactly `n` values.
- Each value finite with `0 <= f_i <= 1e6`.
- `sum_i f_i > 0`.

Any violation scores `Ratio: 0.0`.

## Objective

Minimize the normalized leakage-energy index `E(f)` defined above.

## Scoring

The checker computes your `F = E(f)` and an internal baseline `B = E` of the
**half-block** construction it builds itself: unit gain on the first
`floor(n/2)` sensors, zero on the rest (all gain crammed into one contiguous
half of the array). This is a trivially feasible but deliberately concentrated
allocation. With a minimization normalization:

```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```

Reproducing the baseline scores `~0.1`; a 10x lower index caps the ratio at
`1.0`. Scoring is exact and deterministic.

## Constraints

- `12 <= n <= 30`.
- Exact `O(n^2)` autoconvolution; runs well within the time limit.

## Example

For `n = 4`:

- **Half-block** `f = (1,1,0,0)`: leakage spectrum `c = (1,2,1,0,0,0,0)`,
  `sum_k c_k^2 = 1+4+1 = 6`, `sum f = 2`, so
  `E = 4 * 6 / 2^4 = 1.5` -> the checker's baseline.
- **Uniform** `f = (1,1,1,1)`: leakage spectrum `c = (1,2,3,4,3,2,1)`,
  `sum_k c_k^2 = 44`, `sum f = 4`, so
  `E = 4 * 44 / 4^4 = 0.6875` -> `Ratio = 100 * 1.5 / 0.6875 / 1000 ~= 0.218`.
- A **shaped** profile lowers `E` further, pushing the ratio above `0.22`.
