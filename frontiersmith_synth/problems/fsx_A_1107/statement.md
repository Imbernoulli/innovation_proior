# Uniform-Regime Arithmetic Circuit for a Hypot-Softplus Composite

## Problem
A numerical-kernel library needs to evaluate one fixed composite function

```
f(x) = A*(sqrt(x^2 + B^2) - x) + C*(B / sqrt(x^2 + B^2)) + E*log(1 + exp(D*x))
```

on inputs `x` ranging from `1e-30` to `1e30` in magnitude, **both signs**. The
constants `A, B, C, D, E` are fixed positive (D may be negative) reals given
in the input. Because the library ships as a fixed-function hardware kernel,
`f` must be implemented as a **straight-line arithmetic program**: a
numbered sequence of scalar instructions, each computed once, with no
branches, comparisons, or loops.

The instruction set is fixed: `ADD, SUB, MUL, DIV` (binary) and
`ABS, SQRT, EXP, LOG` (unary). Every instruction has a fixed cost:
`ADD/SUB/MUL/ABS = 1`, `DIV = 3`, `SQRT = 4`, `EXP/LOG = 8`. Your program's
cost is the sum of its instructions' costs. Cheaper is better -- but a
program is only scored if it reproduces `f(x)` correctly at **every**
required `x`, including the extremes, where the algebraically-shortest
formula suffers catastrophic cancellation or overflow (see Feasibility).

## Input (stdin)
```
A B C D E
M
x_1
x_2
...
x_M
```
`A,C,E` in `[0.5,3.0]`, `B` in `[0.6,3.0]`; `D` in `[-2.5,-0.4] ∪ [0.3,2.5]`.
`M` grid points `x_i` (floats, magnitude in `[1e-30, 1e30]`, either sign).

## Output (stdout)
```
K
op_1 arg_1 [arg_2]
...
op_K arg_1 [arg_2]
```
`K` is the instruction count. Each `op_i` is one of the 8 named ops above;
binary ops take exactly 2 args, unary ops exactly 1. Each `arg` is either
the literal token `x`, a finite numeric literal (e.g. `2.5`, `-1.0`,
`1e-3`), or `r<i>` referring to the result of instruction `i` (`1`-indexed,
`i` strictly less than the current line). The value of instruction `K` is
the program's output. `1 <= K <= 400`.

## Feasibility
The program is evaluated in double-precision floating point (IEEE-style:
`exp` overflow -> `+inf`, `sqrt`/`log` domain errors -> `nan`, division by
zero -> `nan`) at every `x_i`. Let `y` be the computed value and `t` be
`f(x_i)` computed to high precision. The output is feasible iff, at **every**
`x_i`:
- `y` is finite (no `nan`/`inf`), and
- `|y - t| <= max(1e-9 * |t|, 1e-300)`.

Any parse error (bad header, wrong token count, unknown op, wrong arity, a
register referencing a non-earlier line, a non-finite or oversized literal)
or any accuracy violation scores `Ratio: 0.0`.

## Objective
Minimize the total instruction cost `cost = sum of per-instruction costs`.

## Scoring
Let `B_ops = 200` (a fixed internal reference cost). With your feasible
program's cost `cost`:
```
Ratio = min(1, 0.1 * B_ops / cost)
```
A correct-but-unrefined program near cost 200 scores about `0.1`; halving
the cost roughly doubles the ratio.

## Constraints
- `1 <= M <= 90`, `1 <= K <= 400`, literal magnitudes `<= 1e18`.
- Time limit 5s, memory 512MB. Deterministic scoring; nothing is timed.

## Example
Toy function `g(x) = 2*x + 3` (illustrative FORM only, not the real `f`).
A feasible 3-instruction program:
```
2
MUL x 2.0
ADD r1 3.0
```
computes `r1 = 2x`, `r2 = 2x+3`; cost `= 1 (MUL) + 1 (ADD) = 2`.
