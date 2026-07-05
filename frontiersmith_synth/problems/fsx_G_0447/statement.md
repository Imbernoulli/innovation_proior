# Common-Subexpression Scheduling for a Multi-Output Polynomial Kernel

## Problem
A compiler backend must emit machine code for a kernel that evaluates **several
target polynomials at once** over a shared set of scalar inputs. Because the
targets are built from overlapping monomials and partial products, a naive
"evaluate each term from scratch" lowering wastes many multiplications. Your
job is to schedule a **straight-line program** (an arithmetic circuit / DAG)
that computes every target *exactly* while using as **few scalar operations**
as possible, by reusing common subexpressions.

You are given `m` target polynomials over input variables `x_0 .. x_{n-1}`.
Each target is a sum of signed monomials (products of input variables). Emit a
straight-line program using only the three scalar binary operations
`ADD`, `SUB`, `MUL` over inputs, integer constants, and previously computed
temporaries, such that each target is produced exactly.

## Input (stdin)
```
n m
```
followed by `m` target blocks. Each block is:
```
K
c d v1 v2 ... vd      (K lines)
```
`K` is the number of terms in the target. Each term line gives an integer
coefficient `c`, a degree `d`, and `d` input-variable indices; the term equals
`c * x_{v1} * x_{v2} * ... * x_{vd}`. The target is the sum of its `K` terms.
All coefficients are non-zero; `0 <= v < n`.

## Output (stdout)
The first line is a single integer `L`, the number of instruction lines that
follow. Then `L` instructions, one per line, in the fixed schema:
```
tK OP A B
```
* `tK` is the destination temporary. Temporaries **must** be defined as
  `t0, t1, t2, ...` in order, each defined exactly once.
* `OP` is `ADD`, `SUB`, or `MUL` (`tK = A OP B`, integer arithmetic).
* `A` and `B` are operands, each one of:
  * `xI` — input variable `x_I`,
  * `tI` — a temporary defined on an earlier line,
  * an integer constant (a bare integer, optionally written `#C`).

After the instructions, emit exactly one final line:
```
OUT o0 o1 ... o_{m-1}
```
where `o_j` is the operand (usually a temporary) whose value equals target
`y_j`.

## Feasibility
The circuit is feasible iff it parses, every temporary is defined once and in
order, every operand references a valid input / earlier temporary / integer
constant, and **each output operand is exactly polynomially equal to its
target** (verified by exact multivariate integer-polynomial arithmetic — no
floating point, no tolerance). Any violation scores `0`.

## Objective
Minimize the number of scalar operations `F` = the number of instruction lines
(each `ADD`/`SUB`/`MUL` counts as one operation). Loading a variable or a
constant is free; only the three binary operations are counted.

## Scoring
Let `B` be the operation count of the grader's naive per-term construction
(rebuild each monomial from scratch, multiply in the coefficient, accumulate).
The score for a feasible circuit is
```
Ratio = min(1, 0.1 * B / F)
```
Reproducing the naive construction scores about `0.1`; a circuit using ten
times fewer operations would saturate at `1.0`. Minimal-operation scheduling of
a shared-subexpression DAG is NP-hard, so the optimum is unknown and there is
ample headroom.

## Constraints
`2 <= n <= 15`, `2 <= m <= 13`, term degree `d <= 5`, coefficients in
`[-3, 3]\{0}`. The straight-line program may use at most `100000` instructions;
intermediate polynomials with more than `200000` monomials are rejected.

## Example (worked score)
Suppose the targets are `y_0 = x0*x1 + x1*x2` and `y_1 = x0*x1 - x1*x2`.
The naive lowering computes `x0*x1` and `x1*x2` **twice** (once per target):
`B = 4` multiplies `+ 2` add/subs `= 6`. A shared circuit
```
4
t0 MUL x0 x1
t1 MUL x1 x2
t2 ADD t0 t1
t3 SUB t0 t1
OUT t2 t3
```
uses `F = 4` operations, so `Ratio = min(1, 0.1 * 6 / 4) = 0.15`. (This tiny
example is illustrative; the graded instances are much larger and reward deeper
sub-product sharing.)
