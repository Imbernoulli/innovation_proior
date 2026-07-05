# Corridor Congestion Constant

## Problem

A linear wildlife corridor is divided into `n` equal, contiguous segments,
indexed `0 .. n-1`. You choose a **habitat-density profile**
`f = (f_0, f_1, ..., f_{n-1})` with each `f_i >= 0`: the amount of resources
(vegetation, water, cover) you place in segment `i`.

Two animals that start in segments `i` and `j` tend to meet somewhere around the
combined location `k = i + j`. The **pairwise encounter pressure** at combined
location `k` is

```
c_k = sum over all i,j with i+j=k of  f_i * f_j        (k = 0 .. 2n-2)
```

i.e. the autoconvolution of the profile with itself. A high `c_k` means a
crowding hotspot. You want to allocate habitat so that the **worst** hotspot is
as small as possible relative to the total population you support.

Define the **normalized peak-congestion index**

```
c1(f) = 2 * n * max_k c_k  /  ( sum_i f_i )^2
```

The index is scale-invariant (multiplying `f` by a constant does not change it),
so only the *shape* of the profile matters. **Minimize `c1(f)`.**

This is a step-function instance of the classical autoconvolution / first
autocorrelation inequality: the uniform profile gives `c1 = 2`, and no profile
can drive it to zero — there is a hard positive floor, and getting near it
requires a carefully shaped, non-obvious profile.

## Input (stdin)

A single integer `n` (`8 <= n <= 26`): the number of corridor segments.

## Output (stdout)

`n` real numbers separated by whitespace: the densities `f_0 .. f_{n-1}`.

## Feasibility

- Exactly `n` values.
- Each value finite with `0 <= f_i <= 1e6`.
- `sum_i f_i > 0`.

Any violation scores `Ratio: 0.0`.

## Objective

Minimize the normalized peak-congestion index `c1(f)` defined above.

## Scoring

The checker computes your `F = c1(f)` and an internal baseline `B = c1` of the
**half-block** profile (unit density in the first `floor(n/2)` segments, zero
elsewhere) that it constructs itself. With a minimization normalization:

```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```

Reproducing the baseline scores `~0.1`; a 10x lower index caps the ratio at
`1.0`. Scoring is exact and deterministic.

## Constraints

- `8 <= n <= 26`.
- Exact `O(n^2)` autoconvolution; runs well within the time limit.

## Example

For `n = 8`:

- **Half-block** `f = (1,1,1,1,0,0,0,0)`: peak `c_k` occurs at `k=6` with value
  `4`, `sum f = 4`, so `c1 = 2*8*4 / 16 = 4.0` -> `Ratio = 0.1`.
- **Uniform** `f = (1,1,1,1,1,1,1,1)`: peak `c_k = 8` at the center,
  `sum f = 8`, so `c1 = 2*8*8 / 64 = 2.0` -> `Ratio = 0.2`.
- A **shaped** profile can reach `c1 ~= 1.59`, giving `Ratio ~= 0.25`.
