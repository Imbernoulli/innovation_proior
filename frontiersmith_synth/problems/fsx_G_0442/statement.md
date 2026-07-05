# Gate-Minimal Circuits for Symmetric Boolean Functions

## Problem
A boolean function `f(x0, x1, ..., x_{n-1})` is **symmetric** when its value depends
only on the *popcount* of the input (the number of bits equal to 1). Such a function is
fully described by a **spectrum**: a string `s` of length `n+1` where `s[k] = 1` iff the
function accepts every input with exactly `k` ones.

Your job is **logic synthesis**: build a combinational circuit that computes `f` using
only 2-input `AND`, 2-input `OR`, and 1-input `NOT` gates, while using **as few gates as
possible**. The inputs `x0..x_{n-1}` are given for free. There is no known closed form for
the minimum gate count of an arbitrary symmetric function, so this is a genuinely
open-ended optimization: sorting networks, adder-tree popcounts, threshold decoders and
their prunings all give different, incomparable trade-offs.

## Input (stdin)
```
line 1:  n                      (8 <= n <= 18)
line 2:  s                      a string of length n+1 over {0,1}; the spectrum
```
`f(x) = 1` iff `s[popcount(x)] == '1'`. The spectrum is never all-0 or all-1, and always
has at least one rejected popcount value.

## Output (stdout)
A straight-line program (a DAG of gates). Wires are indexed as follows:
- wires `0 .. n-1` are the inputs `x0 .. x_{n-1}`;
- the `i`-th emitted gate (0-indexed) defines wire `n + i`.

Print:
```
G                               number of gates (0 <= G <= 2000)
<gate 0>
<gate 1>
...
<gate G-1>
OUT w                           the wire holding f (0 <= w < n+G)
```
Each gate line is one of:
```
AND a b        wire = wire[a] AND wire[b]
OR  a b        wire = wire[a] OR  wire[b]
NOT a          wire = NOT wire[a]
```
Every operand must reference an **earlier** wire: for the gate defining wire `n+i`, each
operand must be strictly less than `n+i` (inputs or previously defined gates only). This
guarantees a valid acyclic evaluation order.

## Feasibility
The output is rejected (score 0) if: the schema is malformed; the gate count is out of
`[0, 2000]`; any operand is a non-integer, out of range, or a forward/self reference;
`OUT` is missing or out of range; or — most importantly — the circuit's truth table does
**not** match `f` on **all** `2^n` inputs. Non-integer tokens (including `nan`/`inf`) are
rejected during parsing.

## Objective (minimize)
Let `F` be the number of gates in your (verified-correct) circuit. The checker builds an
internal baseline `B` = a full bubble sorting network (computing every threshold) followed
by a naive per-value `exactly-k` decoder. The score is
```
Ratio = min(1.0, 0.1 * B / max(1, F))
```
Reproducing the baseline scores `0.1`; a circuit using `10x` fewer gates caps at `1.0`.
Fewer gates is always better.

## Scoring
Deterministic and exact. The checker simulates every wire as a `2^n`-bit integer
(bit-parallel over all inputs), verifies exact equivalence to `f`, then counts gates. No
timing or randomness is involved.

## Constraints
- `8 <= n <= 18`, spectrum length `n+1`.
- `0 <= G <= 2000` gates; each `AND`/`OR` is 2-input, `NOT` is 1-input; each counts as one gate.
- Operands reference strictly earlier wires only.

## Example (worked score)
Suppose `n = 4` and `s = 01110` (accept popcount 1, 2, or 3 — i.e. "not all-equal").
A baseline of `B = 30` gates and your circuit of `F = 12` gates would score
`Ratio = min(1.0, 0.1 * 30 / 12) = 0.25`. (This tiny case is illustrative; the graded
instances use `n >= 8`.)
