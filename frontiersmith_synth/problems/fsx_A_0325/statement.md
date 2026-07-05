# Arena Hype Resonance: Minimizing the First Autocorrelation Constant

## Problem

The grand-final broadcast of an e-sports arena runs for `n` consecutive time-slots.
For each slot `i` you commit an integer **hype intensity** `f[i]` (crowd pyro, LED
surge, caster volume) in `[0, M]`. Not every slot has to fire, but the show cannot
be completely dark.

When the arena replays its own highlight reel, an echo delayed by `k` slots
resonates with the live feed. The strength of the delayed **self-echo** at lag `k`
is the acyclic self-convolution

```
g[k] = sum_i f[i] * f[k-i]          (k = 0 .. 2n-2)
```

A single overwhelming resonance peak blows the mix. Directors therefore measure the
show by its **first autocorrelation constant**

```
c1(f) = 2*n * ( max_k g[k] ) / ( sum_i f[i] )^2 .
```

`c1` is scale-invariant (multiplying every `f[i]` by a constant leaves it unchanged),
so only the *shape* of the intensity vector matters. Your job is to shape the show so
that no single echo lag dominates.

This is a discrete instance of the **first autocorrelation inequality** for
non-negative functions: spreading the mass out lowers the peak, but total mass squared
sits in the denominator, so being too sparse is punished. The exact minimal constant
is not known in closed form; uniform intensity gives `c1 = 2`, yet cleverly shaped
schedules provably do better, and how far below `2` one can push is open.

## Input (stdin)

One line with two integers:

```
n M
```

`n` = number of time-slots, `M` = maximum integer intensity per slot.

## Output (stdout)

Exactly `n` non-negative integers `f[0] f[1] ... f[n-1]`, whitespace-separated
(newlines or spaces both fine), each in `[0, M]`, with `sum f[i] > 0`.

## Feasibility

An output is valid iff it contains **exactly `n`** integer tokens, every token parses
as an integer in `[0, M]`, and at least one is positive. Any other output (wrong
count, out-of-range, non-integer / `nan` / `inf`, all-zero) scores `0`.

## Objective

**Minimize** `c1(f)`.

## Scoring

Deterministic and exact. The checker builds an internal baseline `B` = `c1` of the
naive *front-half block* schedule (uniform intensity in the first `ceil(n/2)` slots,
dark afterwards; `B ~= 4`). With your objective `F = c1(f)`:

```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```

Reproducing the front-half block scores `~0.1`; a uniform show (`c1 = 2`) scores
`~0.2`; driving `c1` toward the conjectured floor keeps climbing. The metric never
saturates on these instances (the constant cannot drop near `B/10`).

## Constraints

- `40 <= n <= 220`, `M = 1000`.
- All scoring is exact integer arithmetic; reruns are bit-for-bit identical.

## Example (worked score)

Suppose `n = 4`, `M = 1000`.

- Baseline front-half block `f = [1, 1, 0, 0]`: `sum = 2`, echoes
  `g = [1, 2, 1, 0, 0]`, peak `2`, so `B = 2*4*2 / 2^2 = 4`.
- Submit uniform `f = [1, 1, 1, 1]`: `sum = 4`, echoes `g = [1, 2, 3, 2, 1]`,
  peak `3`, so `F = 2*4*3 / 4^2 = 1.5`. Score `sc = 100 * 4 / 1.5 = 266.7`,
  `Ratio = 0.2667`.

(The `[1,1,1,1]` example is illustrative of the arithmetic only; at the scored sizes
`n >= 40` uniform yields exactly `c1 = 2`, and beating it requires genuine shaping.)
