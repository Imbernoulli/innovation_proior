# Minimum-Span Non-Redundant Interferometer Track

## Problem
A radio interferometer images the sky by correlating signals from pairs of antennas.
Each pair of antennas produces one *baseline* — a spatial-frequency sample whose value
is the **separation** between the two antennas. Two antenna pairs that share the same
separation measure the *same* spatial frequency: the second measurement is redundant and
wastes an antenna. To sample as many distinct spatial frequencies as possible with `n`
antennas, every pairwise separation must be **distinct** (a *non-redundant* array — the
1‑D case is a Golomb ruler).

All antennas sit on a single straight east–west rail. You choose the integer rail
position (in metres) of each of the `n` antennas. The **span** of the array is the
distance between its westmost and eastmost antenna, i.e. the length of rail you must
build and survey. Longer rails cost more and are harder to keep phase‑stable, so you want
the array as compact as possible while keeping every baseline unique.

Given `n`, place the `n` antennas so that all `C(n,2)` pairwise separations are distinct,
**minimising the span**.

## Input (stdin)
A single integer `n` (`7 <= n <= 18`): the number of antennas.

## Output (stdout)
`n` integers `p_1 p_2 ... p_n` (whitespace-separated, any order): the rail positions in
metres. Each `p_i` must satisfy `0 <= p_i <= 10000000`.

## Feasibility
Your output is accepted only if:
- it contains exactly `n` integers, all finite and in `[0, 10000000]`;
- all `n` positions are **distinct**;
- all `C(n,2)` pairwise separations `|p_i - p_j|` are **distinct** (non-redundant array).

Any violation scores `Ratio: 0.0`.

## Objective (minimise)
`span F = max_i p_i - min_i p_i`.

## Scoring
The checker builds its own feasible reference array `B` (an Erdős–Turán quadratic-residue
ruler on `n` marks) and reports
```
sc = min(1000.0, 100.0 * span(B) / max(1e-9, F))
Ratio: sc / 1000.0
```
Reproducing the reference span scores `~0.100`; a 10×-shorter array caps at `1.0`.
Because optimal non-redundant arrays have no known closed form, real gains come from
search and clever construction.

## Constraints
- `7 <= n <= 18`; positions are integers in `[0, 10000000]`.
- Deterministic scoring; run time `<= 5 s`.

## Example
For `n = 4`, the array `0 1 4 6` has separations `{1,3,4,5,6,2}` — all six distinct — and
span `6`. The array `0 1 2 4` is **rejected**: separations `1` (between 0–1) and `1`
(between 1–2) collide, so the baseline is redundant.
