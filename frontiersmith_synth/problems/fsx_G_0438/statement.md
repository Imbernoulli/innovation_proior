# FIR Filter Kernel: Minimal-Multiplication Schedule for a Fixed Tap Polynomial

## Problem

A stage of a DSP pipeline applies an all-zero (FIR) filter section whose transfer
function numerator is a **fixed monic integer polynomial**

```
H(x) = c[0] + c[1] x + c[2] x^2 + ... + c[d] x^d ,   c[d] = 1
```

where `x` is a symbolic sample value. On the target DSP core / FPGA fabric a scalar
**multiplication** is the scarce resource (each one burns a hardware multiplier /
MAC cycle), whereas additions and subtractions are effectively free. Your job is to
emit a **schedule** (a straight-line program) that evaluates `H(x)` *exactly* using
as few multiplications as possible.

Horner's rule spends `d` multiplications. That is rarely optimal: these numerators
are engineered with hidden algebraic structure, so a cleverer schedule can do far
better. The catch is that only the **dense expanded tap vector** is handed to you —
you must discover any structure yourself.

## Input (stdin)

```
line 1:  d                       (the degree, even, 8 <= d <= 62)
line 2:  c[0] c[1] ... c[d]      (d+1 integer taps, low order first, c[d] = 1)
```

## Output (stdout) — a straight-line program (SLP)

```
line 1:  L                       (number of instructions, 1 <= L <= 512)
next L lines:  <op> <arg1> <arg2>
```

* `op` is one of `mul`, `add`, `sub`.
* each `arg` is one of:
  * `x` — the symbolic sample value;
  * a rational literal, e.g. `-3`, `7`, or `5/2`;
  * `r<i>` — the result of a previous instruction, `0 <= i < ` (current line index).
* Instructions are 0-indexed by their position. The value computed by the program
  is the result of the **last** instruction.

Only `mul`, `add`, `sub` are permitted; there is no division and no other operator.

## Feasibility

The program is **valid** iff every operand is well-formed (finite rational, `x`, or a
backward reference), every intermediate has polynomial degree `<= 2d`, and — as a
polynomial in `x` — it equals `H(x)` **exactly**. Any violation scores `0`.

## Objective (minimize)

`F` = the number of `mul` instructions in your program.

## Scoring

Let `B = d` (the multiplication count of Horner's rule). The checker verifies exact
equivalence, then reports

```
Ratio = min(1, 0.1 * B / F).
```

Reproducing Horner scores `0.1`; a schedule using `10x` fewer multiplications caps
the score at `1.0`. The minimum achievable `F` is genuinely unknown (it involves
shortest addition chains), so headroom always remains.

## Constraints

* `8 <= d <= 62`, `d` even, `c[d] = 1`, taps are exact integers.
* `1 <= L <= 512`.
* Deterministic integer/rational arithmetic only.

## Example (worked score)

Suppose `d = 8` and Horner would use `B = 8` multiplications. A schedule that
evaluates `H` with `F = 4` multiplications scores

```
Ratio = min(1, 0.1 * 8 / 4) = 0.2 .
```

(Illustrative only — not a specific instance below.)
