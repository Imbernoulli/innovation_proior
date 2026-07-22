# Precompensator: Shaping an Input Pulse for a Fixed Filter Cascade

## Problem
A signal you emit is fed through a fixed **cascade** of `K` FIR filters, one after another.
Filter `t` has coefficients `h^(t) = [h^(t)_0, ..., h^(t)_{L_t-1}]`; applying it to a signal
`u` produces the discrete convolution `h^(t) * u`. Passing your input `x` through the whole
chain therefore produces

```
y = h^(1) * h^(2) * ... * h^(K) * x   =   h * x ,
```

where `h = h^(1) * ... * h^(K)` is the combined cascade filter (you must form it yourself).

You are given a **target waveform** `y*`. Craft an **amplitude-bounded input pulse** `x` of
length `N` so that the chain's output matches `y*` closely **without spending much input
energy**. The chain is deliberately ill-behaved: its combined filter has zeros near — and
outside — the unit circle, so naive inversion is treacherous.

## Input (stdin)
```
N A lam K
<for each filter t = 1..K:>
   L_t
   h^(t)_0 h^(t)_1 ... h^(t)_{L_t-1}
M
y*_0 y*_1 ... y*_{M-1}
```
- `N` — length of the input `x` you must output.
- `A` — amplitude bound: every output sample must satisfy `|x_i| <= A`.
- `lam` — the energy weight in the objective (see Scoring).
- `K` filters follow. Let `L = 1 + sum_t (L_t - 1)` be the length of the combined filter `h`.
- `M = N + L - 1` is the length of `y*` (the full convolution length). All values are real.

## Output (stdout)
Exactly `N` real numbers `x_0 ... x_{N-1}` (whitespace-separated), the input pulse.

## Feasibility
`x` must contain exactly `N` finite real numbers with `|x_i| <= A`. Any other output
(wrong count, non-finite value, or an out-of-bound sample) scores `0`.

## Objective (minimize)
Let `y = h * x` (length `M`). Minimize

```
J(x) = || y - y* ||^2  +  lam * || x ||^2 .
```

The first term rewards matching the target; the second penalizes input energy. Because the
cascade has near-unit-circle and non-minimum-phase zeros, an *exact* inverse blows the input
amplitude far past `A`; the amplitude bound and the energy term make this a genuine trade-off.

## Scoring
The checker convolves your `x` through the cascade, computes `J(x)`, and compares it to the
do-nothing baseline `B = J(0) = ||y*||^2` (the objective of the zero input). With
`F = J(x)`, the score is

```
Ratio = min(1000, 100 * B / max(1e-9, F)) / 1000 .
```

The zero input scores `~0.1`; halving `J` relative to a poorly-matched input climbs toward
`1.0`. Ten independent cases are averaged.

## Constraints
`56 <= N <= 196`, `2 <= K <= 6`, small `L_t`, `A = 1`, `lam > 0`. Time limit 5 s, memory 512 MB.

## Example (illustrative)
For a two-tap cascade `h = [1, -1.05]` (a single zero just *outside* the unit circle) and
`N = 4`, the causal inverse `x_n = (y*_n + 1.05 x_{n-1})` grows like `1.05^n` and quickly
exceeds `A` — clipping it then mismatches `y*`. A regularized solve that accepts a small
residual keeps `|x_i| <= A` and scores far better. (Numbers here are only illustrative.)
