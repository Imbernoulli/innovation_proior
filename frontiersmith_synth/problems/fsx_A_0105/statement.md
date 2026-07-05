# Cave Mapping Expedition — Silent Sonar Profile

## Problem
Your expedition is mapping a long, straight cave passage divided into `n` equal segments,
numbered `0 .. n-1`. From a survey sled you emit acoustic pings, one packet of energy per
segment. Let `f[i] >= 0` be the ping energy released while the sled sits in segment `i`.

The rock walls reflect energy, so two packets emitted at segments `i` and `j` produce a
*self-echo* that arrives together at relative delay `k = i + j`. The total self-interference
recorded at delay `k` is the discrete **autoconvolution**

```
g[k] = sum over i+j = k of  f[i] * f[j]        (k = 0 .. 2n-2)
```

A loud coincident echo at any single delay blinds the sonar. You therefore want to shape the
emission profile `f` so that the **peak** self-interference is small relative to the total
energy you spent. The figure of merit (a discrete form of the first autocorrelation
inequality constant `C1`) is

```
C1(f) = 2n * max_k g[k] / ( sum_i f[i] )^2
```

Lower is better. A flat profile (ping equally everywhere) already does reasonably well, but it
is *not* optimal — irregular, carefully balanced profiles suppress the peak further. Finding
the profile that minimizes `C1` is an open extremal problem; you only need to beat the naive
baselines.

## Input (stdin)
A single line with one integer `n` (the number of passage segments).

## Output (stdout)
`n` real numbers `f[0] f[1] ... f[n-1]` (whitespace/newline separated), each the ping energy
for that segment. Every value must satisfy `0 <= f[i] <= 1e9`, and the total energy
`sum f[i]` must be strictly positive.

## Feasibility
The output is rejected (score `0`) unless it contains exactly `n` finite numbers, all within
`[0, 1e9]`, with a strictly positive sum.

## Objective
Minimize `C1(f) = 2n * max_k g[k] / (sum_i f[i])^2`.

## Scoring
The checker builds an internal baseline `B` from a naive "fill the first half of the passage
uniformly, stay silent in the second half" profile and computes its `C1 = B`. Your profile is
scored by

```
Ratio = min(1000, 100 * B / max(1e-9, C1(your f))) / 1000
```

so reproducing the naive half-fill baseline scores `~0.1`, and driving `C1` far below it scores
higher (the score is scale-invariant in `f`). No feasible profile can reach `Ratio = 1`.

## Constraints
- `120 <= n <= 480`.
- Deterministic scoring; exact floating arithmetic with the formula above.

## Example
For `n = 4`, the naive half-fill baseline is `f = [1, 1, 0, 0]`: `g = [1, 2, 1, 0, 0]`,
`max g = 2`, `sum f = 2`, so `B = 2*4*2 / 2^2 = 4`. Submitting the uniform profile
`f = [1, 1, 1, 1]` gives `g = [1, 2, 3, 2, 1]`, `max g = 3`, `sum f = 4`, so
`C1 = 2*4*3 / 4^2 = 1.5` and `Ratio = 100 * 4 / 1.5 / 1000 = 0.2667`.
