# Debris-Sweep Self-Overlap Minimisation

## Problem

An orbital debris-cleanup fleet makes repeated collector passes over a congested
orbital shell. We coarsen one revolution into `n` equal **phase bins** and let
`f = (f_1, ..., f_n)` be the fleet's **sweep-density profile**: `f_i >= 0` is how
much collecting effort is deposited in phase bin `i`.

When two collector passes are separated by a relative phase shift `k`, the amount
of **wasted self-overlap** (two collectors sweeping the same debris at the same
relative geometry, a safety and efficiency hazard) is proportional to the discrete
self-convolution

```
g_k = sum_i f_i * f_{k-i}          (k = 0 .. 2n-2)
```

The most dangerous relative phase is the one that maximises this overlap. We want a
profile whose *worst* relative-phase self-overlap is as small as possible **per unit
of total collecting effort deposited**. The scale-invariant figure of merit is the
**first-autocorrelation constant**

```
c1(f) = 2n * ( max_k g_k ) / ( sum_i f_i )^2 .
```

Minimising `c1(f)` is exactly the discrete form of the classical *first
autocorrelation inequality* for non-negative functions — a constant whose true
minimum is a well-known open research problem (no closed-form optimum is known;
the best rigorous bounds sit near `1.5`). Your job is to construct a good profile.

## Input (stdin)

A single integer `n` (`5 <= n <= 30`), the number of phase bins.

## Output (stdout)

Exactly `n` real numbers `f_1 ... f_n`, whitespace-separated (any layout), the
sweep-density profile. Each `f_i` must be finite, `f_i >= 0`, `f_i <= 1e9`, and the
total `sum_i f_i` must be strictly positive.

## Feasibility

Output is rejected (score `0`) unless it contains exactly `n` tokens, every token
is a finite non-negative number within the cap, and the total mass is positive.
`nan`/`inf` are rejected.

## Objective

**Minimise** `c1(f)`. Scoring is scale-invariant, so multiplying `f` by a positive
constant does not change your score.

## Scoring

The checker computes your `F = c1(f)` and an internal baseline `B = c1(bump)`, where
`bump` is a fixed smooth centred Gaussian profile (a plausible naive "just spread the
mass smoothly" guess). Because this is a minimisation problem the reported score is

```
Ratio = min(1000, 100 * B / F) / 1000 .
```

Reproducing the baseline scores about `0.1`; a profile ten times better than the
baseline caps at `1.0`. Since `c1` cannot fall below the (open) optimum near `1.5`,
the score never saturates — there is genuine headroom for better constructions.

## Constraints

- `5 <= n <= 30`
- `0 <= f_i <= 1e9`, all finite, `sum f_i > 0`
- Deterministic scoring; exact float arithmetic on `c1`.

## Example (worked score)

Suppose `n = 5`. The baseline bump gives `B = c1(bump) ~= 4.047`.

- A **flat** profile `f = (1,1,1,1,1)` has `g_k` peaking at the centre with value `5`
  and total mass `5`, so `c1 = 2*5*5 / 25 = 2.0`. Score `= 100*4.047/2.0 / 1000 ~= 0.202`.
- A carefully shaped profile reaching `c1 ~= 1.64` scores `100*4.047/1.64 / 1000 ~= 0.247`.
- A single spike `f = (1,0,0,0,0)` has `g_0 = 1`, mass `1`, giving `c1 = 10` (much worse
  than the baseline) and a correspondingly low score.

There is no known optimal profile; several distinct strategies (flat, plateau,
boundary-loaded, gradient-optimised) trade off differently as `n` grows.
