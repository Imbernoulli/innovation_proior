# Ridgeline Watchtowers: Sightline-Free Placement of Maximum Value

## Problem
A national forest is mapped onto a combinatorial grid of candidate watchtower cells.
Each cell is addressed by an `n`-digit **terrain code** over the three elevation bands
`{0, 1, 2}` (valley, slope, ridge), so there are exactly `3^n` cells, one per string in
`{0, 1, 2}^n`. Each cell `s` has an integer **surveillance value** `w(s)` in `1..9`
(how much forest that tower can watch over).

Some cells are **blocked** (sheer cliffs, no foundation). You may build a tower on any
subset of the remaining cells, but there is a hazard called a **resonant sightline**:
three *distinct* towers `a`, `b`, `c` such that in **every** digit position `i`,
`(a_i + b_i + c_i) mod 3 == 0`. Three such towers heliograph-flash in a way that blinds
each other, so no such triple may all be built. (Equivalently, `c` is the unique third
cell completing the line through `a` and `b`; this is exactly the **cap set** condition
over `F_3^n`.)

Build a sightline-free set of towers whose total surveillance value is as large as possible.

## Input (stdin)
```
n
b
<b lines, each an n-character terrain code = a blocked cell, ascending>
<one line: 3^n integers = the surveillance values>
```
The final line lists `w` for **every** cell, in **ascending lexicographic order** of the
`3^n` terrain codes (`00..0`, `00..1`, `00..2`, `00..10`, ..., `22..2`). So the
`(k+1)`-th integer is the value of the `k`-th code in that order.

## Output (stdout)
Print your chosen towers, **one terrain code per line** (an `n`-character string over
`{0,1,2}`), and nothing else. The order of lines does not matter; print no count.

## Feasibility
An output is valid iff **all** hold:
- every line is a single string of exactly `n` characters, each in `{0,1,2}`;
- the codes are pairwise distinct;
- no chosen code is a blocked cell;
- no three distinct chosen towers form a resonant sightline.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = sum of w(s)` over the chosen towers `s` (a maximum-value cap set in
`F_3^n` that avoids blocked cells).

## Scoring
Let `B` be the total value of the checker's own reference construction, the
**shoreline-ridge subcube**: every cell whose first digit is `0` and whose remaining
digits are all in `{0,1}`. That set is always sightline-free and never blocked, so it is
always available. With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the shoreline ridge scores `Ratio = 0.1`; a placement worth `10x` the
reference caps at `1.0`.

## Constraints
- `3 <= n <= 7`, so `27 <= 3^n <= 2187`.
- Surveillance values are integers in `1..9`.
- The shoreline-ridge subcube cells are never blocked (the baseline is always available).
- Time limit 5s, memory 512m.

## Example
Suppose `n = 3` with a handful of blocked cells. The shoreline-ridge subcube
`{000, 001, 010, 011}` is sightline-free; if its cells are worth `9+4+7+6` then `B = 26`
and it scores `Ratio = 0.1`. A cleverly chosen sightline-free set worth `F = 67` gives
`sc = 100*67/26 = 257.7`, so `Ratio = 0.257693`.
