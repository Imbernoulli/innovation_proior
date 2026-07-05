# Low-Glare Street Lighting Profile

## Problem
A smart-city district manages a long boulevard divided into **N** consecutive segments.
Segment *i* (0-indexed) carries a streetlamp whose light intensity you may set to any real
value `f_i` with `0 <= f_i <= u_i`, where `u_i` is the hardware/regulatory ceiling for that
segment.

Each lamp's beam optically smears along the boulevard, so the light footprint that actually
lands on the ground is the **self-convolution** of the intensity profile. The pooled
illumination at combined position `k` is

```
G(k) = sum over i+j=k of  f_i * f_j         (k = 0, 1, ..., 2N-2)
```

The city wants a profile that delivers plenty of light but keeps the **worst-case pooled
peak** (the brightest, most glaring spot) small relative to the total light delivered.
Concretely, minimize the normalized peak constant

```
c1(f) = 2*N * max_k G(k) / (sum_i f_i)^2 .
```

Because `c1` is invariant to multiplying every `f_i` by a positive constant, only the *shape*
of the profile matters; the ceilings `u_i` bound the relative heights you may use. Uniformly
lighting every segment gives `c1 = 2`; doing better than that is a genuinely open question
(no exact optimum is known).

## Input (stdin)
```
N
u_0 u_1 ... u_{N-1}
```
`N` is the number of segments (`24 <= N <= 200`). The second line holds the `N` non-negative
ceilings `u_i` (real, one to three decimals).

## Output (stdout)
Print `N` real numbers `f_0 ... f_{N-1}` (whitespace/newline separated), the chosen intensity
for each segment.

## Feasibility
An output is feasible iff:
- it contains exactly `N` finite real numbers,
- `0 <= f_i <= u_i` for every `i` (a tolerance of `1e-6` is allowed on the upper bound), and
- `sum_i f_i > 0`.

Any violation scores `0`.

## Objective
Minimize `c1(f)` (smaller is better — lower peak glare per unit of delivered light).

## Scoring
Let `B` be the constant achieved by the reference *baseline* construction (lamps set to their
ceiling on the first `ceil(N/3)` segments, off elsewhere). Your score is

```
Ratio = min(1.0, 0.1 * B / c1(f)) .
```

Reproducing the baseline scores about `0.1`; every 10x reduction in `c1` relative to `B`
adds toward the cap of `1.0`.

## Constraints
- `24 <= N <= 200`.
- `0 <= u_i <= 2`.
- Time limit 5 s, memory 512 MB.

## Example
Suppose `N = 3` and `u = [1, 1, 1]`.
- Uniform `f = [1, 1, 1]`: `G = [1, 2, 3, 2, 1]`, peak `3`, `sum f = 3`, `c1 = 2*3*3/9 = 2.0`.
- Baseline (`ceil(3/3)=1` segment on) `f = [1, 0, 0]`: `G = [1,0,0,0,0]`, peak `1`, `sum f = 1`,
  `c1 = 2*3*1/1 = 6.0`, so `B = 6.0`.
- Uniform therefore scores `Ratio = 0.1 * 6.0 / 2.0 = 0.3`.
