# Morphogen Golf — the shortest gene that grows the creature

A **creature** is a 2D figure drawn by a *turtle* that reads a string over the alphabet
`F + -` (`F` = step forward one unit drawing an edge; `+` = turn left 90°; `-` = turn
right 90°). Two extra turtle symbols exist for you to use: `[` pushes the turtle state,
`]` pops it. Every other token is a **variable**: invisible to the turtle, but it can be
*rewritten*.

You are given one creature as its developed **transcript** `T` (the exact string the turtle
drew). Your job is to hand back the shortest **gene** — an L-system — whose growth draws the
*same creature*.

## Input (stdin)
```
L
t_1 t_2 ... t_L
```
Line 1 is the length `L`. Line 2 is `T`: `L` space-separated tokens over `F + -`.

## Output (stdout) — an L-system
```
n
axiom tokens...
k
LHS  RHS tokens...        (k lines)
```
`n` is an iteration count (`0 ≤ n ≤ 256`). Growth applies **all** rules simultaneously
`n` times: each token that is a rule's `LHS` is replaced by that rule's `RHS`; a token with
no rule is left unchanged (so `F`, `+`, `-` persist unless you give them a rule — and you
may, exactly as real L-systems rewrite `F`). After `n` steps, the turtle draws the resulting
string. An `LHS` is one token; an `RHS` is zero or more tokens; each rule's `LHS` is unique.

## Feasibility (exact-match constraint)
Let `E` be the set of unit edges your grown creature draws and `E*` the target's. Your
output is **feasible** iff, allowing translation and the 8 square-grid symmetries (rotations
+ reflections), `IoU(E, E*) = |E ∩ E*| / |E ∪ E*| ≥ 0.99`. An infeasible output scores 0.
Expansions exceeding an internal size cap, malformed genes, and empty output score 0.

## Objective (minimize)
Gene length in tokens:
`F = (#axiom tokens) + Σ_rules (1 + #RHS tokens) + 1`.
Smaller genes score higher. The score is a decreasing function of `F`, calibrated against the
**literal baseline** `L` (emitting the transcript verbatim, which is always feasible): the
literal gene scores ≈ 0.1, and shorter recursive genes climb toward — but never reach — 1.

## Why this is hard
The transcript is long and looks like data to copy. But it was *grown*: it is the fixpoint
of an unknown recursion. Copying it (`n = 0`, `axiom = T`) is feasible but enormous. Merely
factoring repeated substrings (a context-free grammar) helps, yet still spends a fresh rule
per level of growth, so it stays proportional to `log L`. The tiny genes live elsewhere:
recover the **rule whose repeated application is the creature**, encode that rule once, and
let `n` do the growing. Different creatures hide different rules — you must read `T` and
discover it, not pattern-match this page.

## Example (illustrative — not a target)
Transcript `F + F - F + F` (`L = 4`). A literal gene is `n=0`, `axiom = F + F - F + F`,
`k=0` → `F = 8`. A gene `n=1`, `axiom = A`, rule `A -> F + F - F + F` grows to the same
walk but is *longer* here — recursion only pays once a creature is big enough to have inner
self-similarity. Real instances are.

## Constraints
Time 5 s, memory 512 MB. `L` up to a few ×10⁴. Scoring is fully deterministic.
