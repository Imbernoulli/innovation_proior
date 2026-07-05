# Quantum Lab Wiring — Minimum Peak Crosstalk of a Coupling Profile

## Problem
A cryostat feed-through carries `n` parallel superconducting control lines laid out
along one axis of a chip. You may choose a non-negative **coupling density**
`f = (f_0, f_1, ..., f_{n-1})`, where `f_i >= 0` is the amount of drive amplitude you
route into wiring cell `i`. When two signals separated by a physical lag interfere,
the induced **crosstalk** at lag `k` is the self-convolution

```
g[k] = sum over i of  f_i * f_{k-i}      (k = 0 .. 2n-2)
```

The dominant coherent-error term of the whole feed-through is the **peak** of this
crosstalk profile, `max_k g[k]`. Because you are free to rescale the overall drive
power, only the *shape* of `f` matters; the physically meaningful, scale-free figure
of merit is the normalized peak crosstalk

```
c1(f) = 2 * n * max_k g[k] / (sum_i f_i)^2 .
```

This is a discrete instance of the classical first **autoconvolution inequality**:
the smallest achievable `c1` is a genuinely open constant (no closed form is known),
so there is no reachable optimum — only better and better wiring profiles.

## Input (stdin)
A single line with one integer `n` (the number of wiring cells).

## Output (stdout)
`n` non-negative real numbers `f_0 ... f_{n-1}` (whitespace-separated, any layout).
At least one must be positive. Values need not sum to anything in particular — the
score is scale-invariant.

## Feasibility
Every `f_i` must be finite and `>= 0`, exactly `n` numbers must be emitted, and the
sum must be strictly positive. Any violation scores `0`.

## Objective (MINIMIZE)
Make the normalized peak crosstalk `c1(f)` as small as possible.

## Scoring
Let `F = c1(f)` for your profile and let `B = c1(b)` for the checker's internal
naive **boundary-loaded** baseline `b_i = 1 + 8*((2*(i+0.5)/n) - 1)^4` (all drive
pushed toward the two edge ports). Because this is a minimization,

```
Ratio = min(1000, 100 * B / F) / 1000 .
```

Reproducing the naive baseline scores about `0.1`; a profile ten times better than
the baseline saturates at `1.0`. Uniform drive (`f_i = 1`) already beats the baseline.

## Constraints
`6 <= n <= 33`. Deterministic exact scoring (double precision, 1e-9 tolerances).

## Example (worked score)
Suppose `n = 6`. The checker's baseline `b` gives `B = c1(b) ≈ 2.972`.
- Submitting the baseline itself gives `F ≈ 2.972`, `Ratio = 100*2.972/2.972/1000 = 0.1`.
- Submitting uniform `f = (1,1,1,1,1,1)`: `g` peaks at the center with value `6`,
  `sum f = 6`, so `F = 2*6*6/36 = 2.0` and `Ratio = 100*2.972/2.0/1000 ≈ 0.1486`.
- A well-shaped profile with `F ≈ 1.84` scores `Ratio ≈ 0.1616`.
