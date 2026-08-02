# Cascading Recovery: Context-Aware Panic-Mode Synchronization

## Problem
A tiny language's parser must survive corrupted token streams. A **program** is
a sequence of **statements**:
- `A ;` — an atom statement
- `( chunk ) ;` — a parenthesized group, where `chunk := item (, item)*` and
  `item := A | ( chunk )`
- `{ stmt* }` — a block
- `F ( chunk ; chunk ; chunk ) stmt` — a for-loop: a 3-clause header, then one
  body statement

Some tokens in the stream are corrupted and printed as `?` — a token that is
**never legal** in any position. When the fixed parser below (embedded in the
checker) meets an unexpected token — a `?`, or a `)`/`}` with nothing open to
close — it is in **panic mode** and must resynchronize by skipping forward.
*Which* token(s) to skip to depends on the **context**: a statement position
(top level or inside `{ }`), a parenthesized group (`( )`), or a for-header
(`F ( ... )`). You choose, once, a **synchronization set** of punctuation
types for each of these three contexts — the same table is then applied to
every program below.

## Recovery rule (fixed, applied by the checker)
On an unexpected token the checker (1) **records an error** at that position,
then (2) skips tokens one at a time — using the *current innermost context's*
sync set — until a token in that set appears (or input ends). On landing:
a `)` or `}` that matches the innermost open construct closes it (consumed);
otherwise the innermost construct is silently **abandoned** (not consumed)
and the same landing token is re-checked against the *parent* context (this
can cascade outward through several levels); a `;` inside a statement or
for-header context is consumed and parsing resumes there; a `,` inside a
parenthesized or for-header context is consumed and parsing resumes there.
**Every token skipped over during recovery is silently discarded** —
including a further `?` (suppressing it) or an opening bracket (forfeiting
its construct without closing it). This is why "always skip to the next `;`"
recovers fast but can miss a second nearby error, or swallow an opening
`{`/`(` whose own closer later shows up as a spurious mismatch.

## Input (stdin)
```
K
tokens of program 1 (space-separated)
...
tokens of program K (space-separated)
```
Tokens are drawn from `A F ( ) { } ; , ?`.

## Output (stdout)
Exactly 3 lines — the sync set for the **statement**, **paren**, and
**for-header** contexts, in that order. Each line is a space-separated list
(possibly empty) of distinct integer codes from `{0,1,2,3,4,5}` meaning
`; , ( ) { }` respectively.

## Feasibility
The output is rejected (score `0`) unless it has exactly 3 lines (an
optional trailing blank line is tolerated) and every token on every line
parses as an integer in `[0,5]` (duplicate codes on one line are silently
merged, not an error).

## Objective (maximize)
Run the recovery rule above, with your 3 sync sets, over every one of the
`K` programs. For each reported error, classify it as a **true positive**
if that exact token position held a `?` in the input, otherwise a
**phantom**. Let `TP` and `FP` be the totals summed over all `K` programs
and `F = TP - FP`. Let `B` be the same quantity computed with **empty**
sync sets in every context (the parser recovers only the very first error
in each program, then gives up on the rest of it — the checker's own
built-in baseline). The score is
```
sc    = min(1000, max(0, 100 * F / B))
Ratio = sc / 1000
```
so the give-up-immediately baseline scores `0.1`; catching most of the
planted errors while keeping phantoms low scores well above it.

## Example
Program: `( ? ) { ; A } A ;` — a paren-group statement whose body is
corrupted, immediately followed by a block, then a trailing statement.
With sync set `{;}` for the paren context: recovery skips past the `)`
hunting for a `;`, lands right after the block's `A`, having swallowed the
`)` and the block's `{`. The block's real `}` then appears with nothing
open to close — a **phantom**. With paren sync set `{,  )}` instead:
recovery finds the `)` immediately (0 tokens skipped), the paren closes
cleanly, and the block parses with no phantom. *(Illustrative shape only —
not the mix of programs in the actual test cases.)*

## Constraints
`1 <= K <= 10` programs per test case, each up to roughly 40 tokens, nested
up to 3 levels deep. All scoring is exact integer bookkeeping over a
single deterministic pass per program — no floating point, no randomness,
no timing.
