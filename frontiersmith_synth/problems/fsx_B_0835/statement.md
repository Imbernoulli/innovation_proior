# Shared-Reagent Triage Program

## Problem

A clinic wants a triage **program**: given a patient's raw lab-measured
features, run a sequence of threshold tests to determine the patient's
category. Each test's input value is not read directly — it must first be
computed by a short **preparation chain** of scalar arithmetic operations
(`ADD`, `SUB`, `MUL`) over the raw features and fixed constants. Different tests' chains can **share instructions**: if test A's chain
already computed some intermediate quantity, and test B's chain needs that
same quantity (the *same* instruction index), B does not pay to recompute
it — the value stays "live" for the rest of that patient's path. Op count
is what you pay for; nothing about time is measured.

You must output a decision **program**: a binary tree whose internal nodes
run a test (paying for any of its prep instructions not yet live on this
path, plus one comparison) and branch on the outcome, and whose leaves
declare a category. The program must be **exactly correct**: for every
patient in the instance, walking the tree from the root must reach a leaf
declaring that patient's given true category. Your objective is to minimize
the **expected** number of operations (instructions + comparisons) paid per
patient, weighted by each patient's given population weight, computed
exactly as a rational number.

Note: some tests are individually very informative yet expensive and share
nothing with any other test; others look less informative alone, share most
of their prep with a sibling test, and together may already fully determine
the category, making any further test pointless even if it looked
attractive alone. A good program reasons jointly about information and
shared cost, not either alone.

## Input (stdin)

```
K M T N
M lines: OP A B          (OP in {ADD,SUB,MUL}; A,B each one of F<i>, I<j>, C<v>)
T lines: FINAL_INSTR THRESHOLD
N lines: F_0 ... F_{K-1} WEIGHT LABEL
```
`F<i>` = raw feature `i` (0-indexed), `I<j>` = the value of instruction `j`
(0-indexed, `j` strictly less than the current instruction's own index),
`C<v>` = the integer literal `v` (may be negative). Instruction `k`'s value
is `OP(resolve(A), resolve(B))`. Test `t`'s outcome on a patient is `1` if
`INSTR[FINAL_INSTR_t] >= THRESHOLD_t` else `0`. `WEIGHT` is a positive
integer (population count); `LABEL` is the patient's true category
(non-negative integer, given).

## Output (stdout)

```
NUMNODES
NUMNODES lines, each one of:
  LEAF <label>
  TEST <testIndex> <loNode> <hiNode>
```
Node `0` is the root. On patient `p` reaching a `TEST` node: if test
`testIndex`'s outcome on `p` is `0`, go to `<loNode>`; if `1`, go to
`<hiNode>`.

## Feasibility

The program must be a finite structure reachable from node 0 without
exceeding a small step cap per patient (so cycles are rejected), all node
and test indices must be in range, and — for **every** patient — the leaf
reached must declare that patient's given true label. Any violation scores
`Ratio: 0.0`.

## Objective & Scoring

Let a patient's **cost** be the number of prep instructions newly computed
along its root-to-leaf path (an instruction already live from an earlier
node on the same path is free) plus one per test node visited. Let `F` be
the population-weighted average cost (exact rational). Let `B` be the cost
of the fixed baseline program that unconditionally runs every test, in the
given input order, on every patient (this is always a valid, if wasteful,
correct program). The checker prints
`Ratio: %.6f` for `min(1000, 100*B/F) / 1000` — a program costing the same
as the baseline scores `0.1`; a program 10x cheaper saturates at `1.0`.

## Constraints

`1 <= K <= 5`, `M <= 200`, `1 <= T <= 8`, `1 <= N <= 200`. Time limit 5s.

## Example (worked, illustrative only)

Tests `t0`(chain len 3, threshold 0), `t1`(chain len 3, sharing all 3 with
`t0`, extra 1 instr) on 2 patients, both needing `t0` and `t1`: baseline
`B` = 4 (union) + 2 (compares) = 6. A program using `t0` then `t1` pays
`3+1` (t0) `+1+1` (t1, only its 1 new instr) `= 6` too here; sharing more of
the chain, or skipping a redundant third test entirely, is what separates a
merely-correct program from a cheap one.
