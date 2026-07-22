# The Conjugated Word Trick: Minimal-Op Reconstruction

## Problem
A magician's black box computes a function `f` on `w`-bit words (all arithmetic
is mod `2^w`, i.e. every intermediate value lives in a fixed-width register).
You are handed the **complete act**: the full truth table of `f`, every input
paired with its output. Your job is to write a short **straight-line program
(SLP)** — a fixed sequence of scalar instructions over registers — that
reproduces `f` **exactly**, using as few operations as possible.

Internally, `f` was built by composing an affine "scramble in" `M1`, one fixed
nonlinear word-trick `T`, and an affine "scramble out" `M2`:
`f(x) = M2(T(M1(x)))`, where `M1(x) = ROTL(x,r1) XOR maskA` and
`M2(z) = ROTL(z XOR maskB, r2)` (`ROTL` = bitwise left-rotation on `w` bits).
`T` is one of a small family of classic bit tricks (e.g. isolate-the-lowest-
set-bit, smear-the-top-bit-downward, broadcast-the-parity-bit, isolate-the-
lowest-zero-bit) — you are not told which one, nor `r1,r2,maskA,maskB`. You
only ever see the finished table. Reproducing it row-by-row always works but
costs many operations; recognizing the conjugated shape lets you rebuild it
in a handful.

## Input (stdin)
```
w
f(0)
f(1)
...
f(2^w - 1)
```
`w` is a small integer; `f(v)` is `f`'s value on input `v`, for `v = 0..2^w-1`
in order, each in `[0, 2^w)`.

## Output (stdout)
```
K
line_1
...
line_K
```
Register `0` is always the input `x` (free, not a line). Lines `1..K` each
define register `i` (the `i`-th line) by ONE instruction, referencing
**strictly earlier** registers (`0..i-1`) only:
```
CONST c          reg[i] = c mod 2^w                  (free: not counted)
ADD a b | SUB a b | MUL a b | AND a b | OR a b | XOR a b     (binary; 1 op)
NOT a                                                  (unary; 1 op)
SHL a c | SHR a c        c in [0,w]; logical shift     (1 op)
```
Every instruction's result is taken mod `2^w` automatically (fixed-width
registers — no separate masking instruction is ever needed). The program's
output is the value of register `K` (the last line).

## Feasibility
Parse strictly (bad opcode / out-of-range or forward register reference /
non-integer operand / wrong token count / `K` outside `[1,300]` all fail).
Evaluate register `K` for **every** `x` in `[0, 2^w)` and require it equal
`f(x)` exactly. Any parse error or a single mismatch scores `Ratio: 0.0`.

## Objective
Minimize the **op count**: the number of lines that are not `CONST` (`CONST`
lines are free, like referencing the input).

## Scoring
Let `B = 7 * 2^w` (the cost of the naive "one equality-indicator term per
table row" construction: for each candidate value `v`, build an exact
`x == v` flag from scratch and add in `f(v)` when it fires). With `ops` your
op count,
```
Ratio = min(1, 0.1 * B / ops)
```
The naive per-row baseline scores `0.1`. Halving your op count doubles the
ratio; reaching a tenth of `B` caps at `1.0`. True minimal op count under
this instruction set is not known to be reached by any of the strategies
above, so headroom remains.

## Constraints
- `w` is small (each test uses a full `2^w`-row table, `2^w <= 64`).
- `1 <= K <= 300`; register/constant values fit comfortably in machine ints.
- Deterministic exact-integer scoring; no timing, no randomness.

## Example
Suppose `w=2` and the table is `f = [2,3,0,1]` (this is a *different*,
illustrative shape — not one of the actual test cases). The 2-op program
`CONST 1` / `XOR 0 1` (i.e. `x XOR 1`) reproduces it exactly: `ops=2`,
`B=7*4=28`, `Ratio = min(1, 0.1*28/2) = 1.0` (capped). A naive per-row
construction for this same table would cost `7*4=28` ops and score `0.1`.
