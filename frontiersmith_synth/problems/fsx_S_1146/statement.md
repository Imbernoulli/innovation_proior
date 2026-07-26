# Engrave the Law on the Smallest Gear Set

## Problem
A gear-cutting workshop has measured a fixed but unknown integer law `F(x,y)`,
defined for all integers `x,y >= 0`. They give you the **complete** value table
of `F` on the small square `0 <= x,y <= 14`. Somewhere beyond that square the
same law keeps governing much larger `x,y` (up into the millions) -- but you
never get to see those larger values directly. Your job is to engrave the law
onto the smallest possible gear train: a short **straight-line arithmetic
program** that reproduces `F` exactly, everywhere, using as few gears
(instructions) as possible.

It is guaranteed that `F` is exactly realizable by a short program in the
instruction set below. It is *not* guaranteed that a construction which merely
reproduces the published square (a lookup, a curve fit, a case-by-case patch)
will keep working past it -- the grader checks far outside the square, so only
the genuine law survives.

*Illustrative FORM only, not the hidden law*: a toy law like `x+y` could be
engraved as `ADD x y` (1 gear). The real `F` is unrelated to this example.

## Input (stdin)
```
testId M
```
then `M+1` lines, each with `M+1` integers: line `x` (0-indexed) lists
`F(x,0) F(x,1) ... F(x,M)`. Always `M = 14`.

## Output (stdout)
```
L
op_1 A_1 B_1
...
op_L A_L B_L
OUT ref
```
`L` (0 <= L <= 200) is the number of instructions. Instruction `i` (1-indexed)
computes register `R_i = op_i(A_i, B_i)` where `op_i` is one of
`ADD SUB MUL DIV MOD` (integer `+ - * //` `%`, floor-division convention).
Each operand `A_i, B_i` (and the final `ref`) is one of:
- `x` or `y` (the two inputs),
- an integer literal with `|value| <= 1000`,
- `R_j` referring to an earlier instruction's register, `1 <= j < i` (for
  `ref`, `j <= L`) -- no forward references, no cycles.

`DIV`/`MOD` by a divisor that evaluates to `0` are infeasible. Any intermediate
value whose magnitude exceeds `10^30` is infeasible.

## Feasibility
Parse strictly: wrong token counts, unknown opcodes, out-of-range literals,
undefined registers, or non-finite tokens all score `0`. Then, in exact integer
arithmetic (no floats, no tolerance): the program must reproduce **every**
published table entry, **and** every point of a private held-out set (chosen
far outside the `14x14` square, unknown to you, regenerated deterministically
by the grader for this `testId`). Any single mismatch anywhere -> `Ratio: 0.0`.

## Objective
Minimize `L`, the instruction count, subject to exact global correctness.

## Scoring
Let `B = 16` be the grader's reference instruction budget (roughly what a
direct, unsimplified translation of a textbook expansion needs). With your
instruction count `L`:
```
Ratio = min(1, 0.1 * B / L)
```
Matching the reference budget scores `0.1`; a circuit a third that size scores
`0.3`; nothing can exceed `1.0`. The true minimal `L` for this law is not
revealed and is believed to sit below what any of the reference solutions
below reach -- headroom remains above every one of them.

## Constraints
- `0 <= x, y` in the published table; held-out `x,y` can reach `10^6`.
- `0 <= L <= 200`; `|literal| <= 1000`.
- Time limit 5s, memory 512MB, deterministic (same submission -> same score,
  forever).

## Example
Suppose (hypothetically) that some other, unrelated law `G(x,y) = x*y`
reproduced its own tiny published table. The program:
```
1
MUL x y
OUT R1
```
uses `L = 1` gear, well under the reference budget of `16`, so if it also
survived the held-out check it would score `min(1, 0.1*16/1) = 1.0` (capped).
A program using exactly `16` gears that also survives the held-out check would
score `0.1`. A program that reproduces the published square but drifts on the
held-out square -- however small `L` is -- scores `0.0`.
