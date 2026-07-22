# Ambigram Matrix: Center Fixed-Points and the Rotation-Compatibility Graph

## Problem
You are given a **glyph dictionary** of `L` bitmap letters, each a `7x5` binary
grid (7 rows, 5 columns, flattened row-major into 35 bits), and must print a
**single `7x(5k)` binary grid** `M` — `k` glyph *slots* side by side.

`M` is read two ways. **Upright**: split `M` into its `k` slots (columns
`[5(j-1),5j)`) and decode each against the dictionary. **Rotated**: rotate the
*whole* grid `180°` (reverse row order, reverse each row's character order),
split the result into `k` slots the same way, and decode each of those too.
A slot decodes to letter `w` iff `w` is the **unique nearest** dictionary
glyph by Hamming distance, within distance `d`, with a strictly smaller
distance than every other glyph (`nearest < second-nearest`). The
**legibility margin** of a valid decode is `second-nearest - nearest`.

Because `k` is always odd, one slot — the **center** — maps to itself under
the 180° rotation: its rotated decode is literally `decode(rot180(own
bitmap))`. For the grid to be valid, the center's upright letter must equal
its own rotated letter: a genuine **near-fixed-point** of the rotation
(rare — most letters do not look like any letter, themselves included, upside
down). Off-center slots have no such self-constraint, but each slot's
upright letter and its rotated letter are still tied together through the
*same* physical bitmap, so a good slot must be **rotation-compatible**: close
to one dictionary letter upright and close to another (or the same) once
flipped.

## Input (stdin)
```
L k d bonus
<7 lines of 5 chars '0'/'1' : glyph 0>
...
<7 lines of 5 chars '0'/'1' : glyph L-1>
```
`k` is always odd. `bonus` is the per-distinct-letter reward (see Scoring).

## Output (stdout)
Exactly 7 lines, each a string of `5k` characters in `{0,1}`: the grid `M`.

## Feasibility
Invalid (`Ratio: 0.0`) unless **all** hold:
- exactly 7 output lines, each of length exactly `5k`, chars in `{0,1}`;
- every one of the `k` upright slots decodes validly (unique nearest `<= d`);
- every one of the `k` rotated slots decodes validly the same way;
- the center slot's upright letter equals its rotated letter.

## Objective
Let `a_1..a_k` be the upright-decoded letters and `b_1..b_k` the
rotated-decoded letters, with margins `mu_j`, `mb_j`. Maximize
```
F = sum(mu_j) + sum(mb_j) + bonus * |{a_1..a_k} union {b_1..b_k}|
```
— total legibility margin across both readings, plus a bonus for how many
*distinct* letters the ambigram legibly uses in either orientation.

## Scoring
The checker builds its own baseline `B`: repeat the dictionary's first
self-compatible (palindromic) letter in every slot (always valid). With
`sc = min(1000, 100*F/B)`, print `Ratio: sc/1000` — matching `B` scores
`0.1`; `10x` better caps at `1.0`.

## Constraints
`8 <= L <= 20`, `3 <= k <= 11` (odd), `1 <= d <= 20`, `1 <= bonus <= 100`.
Time limit 5s, memory 512MB.

## Example (illustrative only — not a real dictionary)
`L=2`, `k=3`, `d=15`, `bonus=22`. Glyph 0 = `A` (a symmetric "O"-like
palindrome, `rot180(A)==A`). Glyph 1 = `B` (asymmetric), with
`dist(A,B)=12`, `dist(rot180(B),A)=12`, `dist(rot180(B),B)=16`.

Filling all 3 slots with `A`: every upright and rotated decode is `A` at
margin `12` (nearest=self at 0, second-nearest=12), so `F = 6*12 + 22*1 =
94`. This reproduces the checker's baseline exactly, `B=94`, giving
`Ratio=0.1`.

Using `B, A, B` (center still `A`, its only valid self-compatible option):
upright decodes are `B,A,B` at margins `12,12,12` (sum `36`); the rotated
grid's slots are `rot180(B), rot180(A), rot180(B)` (slot order reverses but
slots 1 and 3 are both `B` so this doesn't matter here), decoding to
`A,A,A` at margins `4,12,4` (sum `20`, since `dist(rot180(B),A)=12 <
dist(rot180(B),B)=16`). The center still matches (`A == A`). Distinct
letters used `= {A,B}`, so `F = 36+20+22*2 = 100`, `Ratio = 100*100/94/1000
= 0.1064` — a small, honest improvement from using a second, distinct
letter off-center.
