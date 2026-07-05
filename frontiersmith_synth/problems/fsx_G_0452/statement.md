# Integer MIMO Precoder: Minimal-Operation Linear Straight-Line Program

## Problem
A fixed integer **MIMO precoder** is a matrix `M` with `m` rows (antenna ports) and `n`
columns (data streams). For an input symbol vector `x = (x_0, ..., x_{n-1})` the precoder
must produce the antenna drive vector `y = M x`, i.e. each output

```
y_i = sum_{j=0}^{n-1} M[i][j] * x_j        (exact integer linear form)
```

Because the same precoder is applied to a stream of symbol vectors, the hardware wants to
compute **all `m` linear forms at once** using the fewest scalar operations. The only
primitive operations available are:

- `ADD i j`  : a new value `r = r_i + r_j`
- `SUB i j`  : a new value `r = r_i - r_j`
- `DBL i`    : a new value `r = 2 * r_i`   (a wired left-shift; still counted)

There are **no general multipliers** — every coefficient of `M` must be synthesised from the
inputs using only additions, subtractions, and doublings, freely sharing intermediate
results across outputs. Your job is to emit such a **linear straight-line program** (SLP).

## Input (stdin)
```
m n
M[0][0] M[0][1] ... M[0][n-1]
...
M[m-1][0] ... M[m-1][n-1]
```
All `M[i][j]` are integers.

## Output (stdout)
A straight-line program over registers. Registers `0 .. n-1` are pre-loaded with the inputs
`x_0 .. x_{n-1}`. Instruction number `t` (0-indexed) creates register `n + t`.

```
P
<instr 0>
<instr 1>
...
<instr P-1>
o_0 o_1 ... o_{m-1}
```
where `P` is the number of instructions, each `<instr>` is one of
```
DBL a
ADD a b
SUB a b
```
with `a, b` register indices that are **already defined** (strictly less than the index of
the register being created), and the final line lists the `m` register indices holding
`y_0 .. y_{m-1}`.

## Feasibility
The program is feasible iff every operand references an already-defined register, every
output index is a valid register, and — evaluating each register as an exact integer linear
combination of the inputs — output register `o_i` equals row `i` of `M` **exactly**. Any
parse error, out-of-range index, non-integer / non-finite token, or mismatch scores 0.

## Objective (minimize)
`F` = total number of instructions `P` (every ADD/SUB/DBL costs 1). Fewer is better.

## Scoring
Let `B` be the operation count of the reference **binary double-and-add** construction
(each linear form built independently by the standard binary method). The score is
```
Ratio = min(1.0, 0.1 * B / F)
```
Reproducing the reference costs `Ratio = 0.1`; a 10x reduction saturates at `1.0`.

## Constraints
`1 <= m, n <= 16`, `|M[i][j]| <= 63`, `0 <= P <= 500000`.

## Example
For `M = [[3]]` (m=n=1), the reference binary method builds `3*x_0` as
`DBL 0` (register 1 = `2 x_0`), `ADD 1 0` (register 2 = `3 x_0`), so `B = 2`.
A submission
```
2
DBL 0
ADD 1 0
2
```
outputs register `2 = 3 x_0`, is correct, uses `F = 2`, and scores `Ratio = 0.1`.
This illustrates the *format only*; real instances are dense integer matrices where sharing
power-of-two multiples across the `m` outputs — and signed-digit recoding of the coefficients
— cut the operation count well below the binary reference.
