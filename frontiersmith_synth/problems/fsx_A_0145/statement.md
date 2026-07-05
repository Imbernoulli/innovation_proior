# Damping Bus Resonance in a Solar Inverter Array

## Problem
A solar farm feeds a shared AC bus through a string of `n` inverters. Over one
switching cycle each inverter injects a non-negative current pulse; slot `i`
carries amplitude `f_i`. When these pulses beat against one another the resonant
energy that accumulates on the bus at relative delay `k` is the **self-convolution**

```
(f * f)[k] = sum_{i + j = k} f_i * f_j ,      k = 0 .. 2n-2 .
```

The worst-case resonance is the peak of this profile, `max_k (f*f)[k]`. The total
delivered power is `(sum_i f_i)^2`. Engineers care about the peak resonance *per
unit delivered power*, captured by the scale-invariant first-autocorrelation
constant

```
c1(f) = 2 * n * max_k (f * f)[k] / (sum_i f_i)^2 .
```

Your task: design the emission step vector `f` that makes `c1(f)` as **small** as
possible. A perfectly flat drive gives `c1 = 2`; a smarter shape does better, but
there is no closed-form optimum -- it is an open functional-inequality question.

## Input (stdin)
One line:
```
n V
```
`n` = number of slots (length of the vector); `V` = maximum integer amplitude per slot.

## Output (stdout)
`n` non-negative integers `f_0 .. f_{n-1}` separated by whitespace.

## Feasibility
* exactly `n` integer tokens;
* `0 <= f_i <= V` for every `i`;
* `sum_i f_i > 0` (the profile must not be all zero).

Any violation scores `0`.

## Objective
Minimise `c1(f)`. Because `c1` is homogeneous of degree 0, only the *shape* of
`f` matters, not its magnitude -- use the integer range to approximate any real
profile.

## Scoring
The objective is computed with exact rational arithmetic. The checker builds an
internal baseline `B = c1(triangle)`, where the triangle profile is
`t_i = min(i+1, n-i)`. Your score is

```
sc    = min(1000, 100 * B / c1(f))
Ratio = sc / 1000        (reported per test; higher is better)
```

Matching the triangle baseline scores about `0.1`; driving `c1` ten times below
the baseline caps the score at `1.0`.

## Constraints
`6 <= n <= 64`, `V = 1000`. Ten test cases of increasing `n`.

## Example
For `n = 4`, the flat vector `f = [1, 1, 1, 1]` has `sum = 4`, self-convolution
`[1,2,3,4,3,2,1]` with peak `4`, so `c1 = 2*4*4/16 = 2.0`. The triangle
`t = [1,2,2,1]` has `sum = 6`, self-convolution `[1,4,8,10,8,4,1]` with peak `10`,
so `c1 = 2*4*10/36 = 2.222`. Emitting the flat vector against this baseline scores
`Ratio = 2.222 / 2.0 / 10 = 0.111`.
