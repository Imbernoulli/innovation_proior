# Depressed-Square Evaluator: Minimum-Multiplication Straight-Line Programs

## Problem
You are given ONE fixed polynomial `P(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n` of even
degree `n`, with integer coefficients. Output a **straight-line program** over `{+,-,*}`
that computes `P(x)` exactly, for every value of the runtime variable `x`, using as **few
multiplications** as possible. Additions/subtractions are free; only `*` is charged.

Register `0` is predefined as `x`. Each line of your program creates one new register
(numbered `1, 2, 3, ...` in order) via one instruction:
```
C p q     new register = p/q            (a rational constant; q > 0)
A i j     new register = reg[i] + reg[j]
S i j     new register = reg[i] - reg[j]
M i j     new register = reg[i] * reg[j]     <-- counts toward your score
```
`i` and `j` must be indices of registers already defined (`0` is always valid). The result
of your program is the **last register you define** (or register `0` if you emit zero
instructions). It must equal `P(x)` identically as a polynomial — for every `x`, not just
sampled points.

## Input (stdin)
```
n
a_0 a_1 a_2 ... a_n
```
`n` is even, `4 <= n <= 24`, `a_n != 0`. Coefficients fit comfortably in signed 64-bit
(magnitude well under `10^9`).

## Output (stdout)
```
L
line_1
line_2
...
line_L
```
`L` (the instruction count) on the first line, then exactly `L` instruction lines as above.

## Feasibility
The checker expands your program **symbolically** as an exact polynomial in `x` (rational
coefficients, big integers — never floating point, never sampled) and compares it
coefficient-by-coefficient against the target `P`. Any violation scores `Ratio: 0.0`:
- malformed line / unknown opcode / wrong token count,
- a register index that is out of range or not yet defined,
- a non-finite or unparseable constant, or `|numerator|`/`|denominator| > 10^18`,
- `L > 500` (instruction budget) or `q <= 0` in a `C` line,
- the resulting polynomial does not equal `P(x)` exactly (wrong degree or any
  mismatched coefficient).

## Objective
Minimize `F`, the number of `M` instructions actually executed.

## Scoring
Let `B = 2n - 1` — the cost of the checker's own naive construction (build the power chain
`x^2, x^3, ..., x^n` by repeated multiplication by `x`, `n-1` multiplies, then scale each
of the `n` non-constant monomials `a_i x^i` by its coefficient, `n` more multiplies).
With your multiplication count `F`:
```
Ratio = min(1, 0.1 * B / F)
```
So the naive power-and-scale construction scores `0.1`. Horner's rule — evaluate
`(...((a_n x + a_{n-1}) x + a_{n-2}) x + ... ) x + a_0` — needs only `n` multiplications and
clearly beats the naive baseline, but it treats every coefficient as an opaque number and
is **not** the best you can do: the specific numeric values of `a_0..a_n` were not chosen
uniformly at random, and a construction that *reads* those values before deciding how to
multiply can do substantially better than Horner on every instance in this problem's test
suite. The true minimum multiplication count for a fixed general polynomial is a deep open
question in algebraic complexity — this checker's ceiling is a generous target, not the
proven optimum, so real headroom above what any fixed recipe achieves remains open.

## Constraints
- `4 <= n <= 24` (even), `L <= 500` per submission, time limit 5s.
- Deterministic, exact rational scoring; the checker never samples or times anything.

## Example
For `P(x) = x^2 + 2x + 1` (`n=2`, illustrating the FORMAT only — real instances have
`n >= 4`): the program `1: C 1 1` (register1 = 1), `2: A 0 1` (register2 = x+1),
`3: M 2 2` (register3 = (x+1)^2) computes `P(x)` exactly with `F=1` multiplication,
because `x^2+2x+1` happens to factor as `(x+1)^2` — this is exactly the kind of
value-dependent shortcut Horner's rule can never find.
