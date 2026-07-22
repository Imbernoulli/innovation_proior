# Shared Sub-Plan Compiler: One Circuit for Eight Analysts

## Problem
A data engine serves **K = 8 analysts**. Each analyst `q` has a fixed *query* — a
homogeneous quadratic function of the same `n` shared input columns
`x_0, …, x_{n-1}`:

```
Q_q(x) = sum over monomials  coef * x_i * x_j        (0 <= i <= j < n)
```

You must compile **one** straight-line arithmetic program (an SLP over `+`, `-`, `*`)
that computes **all eight** query outputs, using **as few scalar operations as
possible**. Every instruction — one `+`, one `-`, or one `*` — costs exactly one
operation; wiring an already-computed value into a later instruction or into an
output is free. The eight queries are not random: they secretly share structure, and
the win comes from a plan that computes shared quantities *once* for every analyst.

## Input (stdin)
```
n K
```
then, for each of the `K` queries in turn:
```
T_q
i_1 j_1 c_1
...
i_{T_q} j_{T_q} c_{T_q}
```
`T_q` is the number of nonzero monomials of query `q`; each line gives an integer
coefficient `c` of the monomial `x_i * x_j` (with `i <= j`; `i == j` means `x_i^2`).
Constraints: `1 <= n <= 64`, `K = 8`, all `|c|` small integers.

## Output (stdout)
A straight-line program as a flat whitespace-separated token stream:
```
P
OP_0 A_0 B_0   OP_1 A_1 B_1   ...   OP_{P-1} A_{P-1} B_{P-1}
G_0 G_1 ... G_{K-1}
```
`P` is the number of instructions. Instruction `t` is `OP A B` with `OP` in
`{ +, -, * }`; it defines register `r{t}` = `A OP B`. Each operand `A`, `B`, and each
output operand `G_q` is one token:

- `xI` — input variable `I` (`0 <= I < n`),
- `rJ` — a register **already defined** (`J < t` inside instruction `t`; `J < P` for outputs),
- a **bare rational literal**: integer `-3`, decimal `0.5`, or fraction `7/4`.
  Scientific notation and the tokens `nan`/`inf` are rejected.

`G_q` names the value that must equal `Q_q`. Registers are defined strictly in order
(no forward references). `1 <= P <= 20000`.

## Feasibility
The checker verifies **exact polynomial equivalence**: each output value must equal
its target query `Q_q` as a polynomial (tested at deterministic evaluation points over
a large prime field). Any parse error, wrong token count, forward/undefined register,
non-finite or scientific literal, or a query whose output does not match scores `0`.

## Objective
Minimize `P`, the total instruction (scalar-operation) count.

## Scoring
Let `B = sum_q (3*T_q - 1)` be the operation count of the **naive plan** (for each
query, each monomial: `x_i*x_j`, times its coefficient, then accumulate). With your
count `P`,
```
Ratio = min(1, 0.1 * B / P)
```
The naive plan scores `0.1`. Halving the op count doubles the ratio; reaching a tenth
of `B` caps at `1.0`. The true minimum-size circuit is unknown and lies well below any
straightforward construction, so headroom remains above every reference plan.

## Constraints
- `1 <= n <= 64`, `K = 8`, coefficients are small integers, `1 <= P <= 20000`.
- Deterministic exact scoring (finite-field identity testing); nothing is timed.

## Example
Two analysts (`K = 2`), `n = 2`, with
`Q_0 = x_0^2 + 2 x_0 x_1 + x_1^2` and `Q_1 = 4 x_0^2 + 4 x_0 x_1 + x_1^2`.
Here `B = (3*3-1) + (3*3-1) = 16`. A plan that first computes the shared linear forms
`s = x_0 + x_1` and `t = 2 x_0 + x_1`, then `Q_0 = s*s`, `Q_1 = t*t` uses `P = 4`
instructions and scores `min(1, 0.1*16/4) = 0.4` — because both queries factor through
the *same* small set of linear forms, sharing beats optimizing either query alone.
