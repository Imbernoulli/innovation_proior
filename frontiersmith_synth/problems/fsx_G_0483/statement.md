# Sonar Costas Waveform: Minimum-Coincidence Frequency-Hop Permutation

## Problem
A frequency-hopping sonar transmits, over `n` equal time slots, a pulse at one of `n` distinct
frequencies. The waveform is described by a permutation `p` of `{0, ..., n-1}`: in time slot `c`
the pulse uses frequency `p[c]`. Equivalently, the time-frequency plane is an `n x n` grid with a
single mark in every row and every column.

The **ambiguity performance** of the waveform is governed by its displacement vectors. For any two
time slots `i < j`, the pair contributes the displacement vector

```
( j - i ,  p[j] - p[i] )        (time offset, frequency offset)
```

A vector that occurs for **more than one** pair is a **coincidence**: those pairs are
indistinguishable in the sonar's ambiguity function, producing spurious sidelobes. A waveform whose
`C(n,2)` displacement vectors are all distinct is a **Costas array** (a "thumbtack" ambiguity
function with no coincidences).

Perfect Costas arrays are not available for every order, and are not reachable by any cheap formula
here: this problem uses orders `n` for which **neither** the Welch (`n+1` prime) **nor** the
Lempel-Golomb (`n+2` a prime power) algebraic construction applies. Two ladder orders (`n = 32` and
`n = 33`) are proven to admit **no** Costas array at all, so the minimum achievable coincidence
count there is genuinely open. Your job is to build a permutation with **as few coincidences as
possible**.

## Input (stdin)
A single integer:
```
n
```
the order of the array (the number of time slots and of frequencies).

## Output (stdout)
`n` integers, whitespace-separated: the permutation `p[0], p[1], ..., p[n-1]`, where `p[c]` is the
frequency used in time slot `c`. The values must be a permutation of `{0, ..., n-1}`.

## Feasibility
The artifact is rejected (score `Ratio: 0.0`) unless it is **exactly `n` integers**, each in
`[0, n-1]`, forming a genuine permutation (each frequency used once). Non-integer, non-finite
(`nan`/`inf`), out-of-range, missing, duplicated, or extra tokens all score `0.0`.

## Objective (minimize)
Let `F` be the number of **coincidences**: over all displacement vectors that occur, sum
`(multiplicity - 1)`. Equivalently `F = C(n,2) - (number of distinct displacement vectors)`.
Smaller `F` is better; a Costas array has `F = 0`.

## Scoring
The checker builds its own trivial baseline `B`: the **identity permutation** `p[i] = i`, whose
displacement vectors are all of the form `(h, h)` (vector `(h,h)` occurring `n-h` times), giving
`B = (n-1)(n-2)/2` coincidences. With your coincidence count `F`,
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
So the identity baseline scores about `0.1`, and a waveform with one tenth as many coincidences (or
fewer) caps at `1.0`. Because the true minimum is unknown for the larger orders, strong strategies
still leave headroom.

## Constraints
`19 <= n <= 54` across the difficulty ladder. Each case runs in well under the time limit.

## Example
Illustrative small case (`n = 4`, NOT a ladder order). The permutation `p = [0, 2, 3, 1]` has marks
at `(0,0),(1,2),(2,3),(3,1)`. Its displacement vectors for the six pairs are
`(1,2),(2,3),(3,1),(1,1),(2,-1),(1,-2)` -- all distinct, so `F = 0` coincidences (a Costas array of
order 4). The identity baseline for `n = 4` is `B = (3)(2)/2 = 3`, so this artifact would score
`min(1000, 100*3/1e-9)/1000 = 1.0`, while the identity itself scores `100*3/3/1000 = 0.1`. For the
larger ladder orders no such perfect array is cheaply available.
