# Gate Diet: the Smallest Circuit for a Badge Predicate

## Problem
A door controller tests `n`-bit badge codes against a tiny protocol state
machine: it starts in state `q0`, reads the code's bits one at a time (bit `0`
first, then bit `1`, …, bit `n-1`), follows the matching transition, and
unlocks iff some run ends in an accept state. The machine may be
**nondeterministic** (one `(state, bit)` pair can list several targets), so a
code is accepted iff *at least one* run accepts.

You must synthesize this unlock predicate as a **straight-line Boolean
circuit** over the `n` input bits, using as few gates as possible. The judge
checks your circuit against the machine's language on **every** one of the
`2^n` codes (exact truth-table equivalence), then counts your gates.

## Input (stdin)
```
n S q0 K M
a_1 a_2 ... a_K
p b q            (M such lines)
```
`n` = code length, `S` = number of states (numbered `0..S-1`), `q0` = start
state, `K` = number of accept states `a_i` (distinct), `M` = number of
transition lines; a line `p b q` means "from state `p`, on bit `b` (0 or 1),
one possible next state is `q`". Several lines may share the same `(p, b)`.
The machine reads bit `0` of the code first.

## Output (stdout)
```
G
<G gate lines>
OUT w
```
Wires: inputs are wires `0..n-1` (wire `i` = bit `i` of the code being
tested). The `g`-th gate line (0-indexed) creates wire `n+g` and may reference
only **strictly earlier** wires. A gate line is one of:
```
CONST c        (c is 0 or 1)
NOT a
AND a b        OR a b        XOR a b
```
`OUT w` names the single output wire (`0 <= w < n+G`), which must carry `1`
exactly for accepted codes. `0 <= G <= 400000`.

## Feasibility
The judge evaluates your circuit on all `2^n` input assignments with exact
Boolean semantics and compares the complete truth table against the machine's
accepted language. Output is feasible iff the two tables are **identical**.
Any parse error, unknown opcode, out-of-range/forward wire reference, bad
literal, trailing token, or truth-table mismatch scores `0`.

## Objective
Minimize `G`, the number of gate lines.

## Scoring
Let `A` be the number of accepted codes and `z(v)` the number of 0-bits of
code `v`. The judge internally builds the naive **sum-of-products** circuit —
one independent product term per accepted code, with a fresh `NOT` gate for
every 0-bit of every code, an `n-1`-gate AND chain per code, and `A-1` OR
gates to combine them — whose exact gate count is
```
B = sum_{v accepted} z(v)  +  A*(n-1)  +  (A-1)
```
Your score is
```
Ratio = min(1, 0.1 * B / max(1, G))
```
Reproducing the naive per-string expansion scores about `0.1`. Two-level DNF
minimization (shared negations, merged product terms) does noticeably better
but stays two-level. The real headroom is **multi-level sharing**: the
machine has far fewer live state-sets per bit position than the language has
strings, and a circuit whose subexpressions mirror that state structure
reuses one subcomputation for exponentially many accepted codes. The exact
minimum gate count is unknown; scores above every reference construction are
achievable.

## Constraints
- `1 <= n <= 13`, `1 <= S <= 64`, `1 <= K <= S`, `0 <= M <= 8192`.
- The accepted language is never empty and never all `2^n` codes.
- Time limit 5 s, memory 512 MB. Deterministic exact scoring; nothing is
  timed or randomized.

## Example
`n = 4`, `S = 4`, `q0 = 0`, accept state `{2}`, transitions `0:0→0,1→1`,
`1:0→1,1→2`, `2:0→2,1→3`, `3:0→3,1→3` — the machine accepts codes with
**exactly two 1-bits**: `0011, 0101, 0110, 1001, 1010, 1100` (bit 0
leftmost), so `A = 6`. The baseline is `B = 12 + 6*3 + 5 = 35` gates (each
accepted code has two 0-bits). A participant circuit is feasible iff it
outputs `1` on exactly those 6 codes. Emitting the naive 35-gate expansion
scores `0.100`; a two-level minimized DNF needs 27 gates (`Ratio ≈ 0.130`); a
circuit sharing the machine's state structure needs 20 gates (`Ratio =
0.175`). Better circuits score higher, up to the cap at `1.0`.
