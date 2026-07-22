# Dead-Comparator-Proof Sorting Network

You are designing sorting hardware from **comparators**. A comparator on wires
`(i, j)` with `i < j` reads the two values, writes the smaller to wire `i` and the
larger to wire `j`. A *comparator network* is a fixed sequence of such comparators;
it is a **sorting network** if, for every input, the wires end in non-decreasing
order (wire `0` smallest).

One comparator on the board may burn out. A burnt comparator simply passes its two
values through unchanged (as if it were deleted). You must build a network that
still sorts **even if any single comparator is removed** — a *1-fault-tolerant*
sorting network — using **as few comparators as possible**.

## Input (stdin)
One line with two integers `n` and `S`: the number of wires and the size
`S = |Batcher(n)|` of the judge's reference sorting network (Batcher odd-even
mergesort). You only need `n`; `S` is provided so you can reason about the score.
`3 ≤ n ≤ 12`.

## Output (stdout)
```
F
i_1 j_1
...
i_F j_F
```
`F` = number of comparators, then one comparator per line with `0 ≤ i < j < n`.

## Feasibility (equivalence gate)
By the **0/1 principle**, a network sorts all inputs iff it sorts all `2^n` binary
inputs; the judge checks this bit-parallel. Your submission is accepted **only if**:
1. the network **sorts** with no fault, and
2. the **single-comparator deletion sweep** holds — for *every* one of your `F`
   comparators, deleting just that comparator still leaves a network that sorts.

If either fails, the score is `0`.

## Objective and Scoring (minimize F)
Let `S` be the reference sort size above and `E = F − S` your redundancy beyond a
plain sort. The score is
```
ratio = min(1,  0.2 · S / max(1, E))     (ratio = 1 if E ≤ 0)
```
Fewer comparators ⇒ larger `ratio`. Reference points: **triplicating** a sorting
network (`F = 3S`, `E = 2S`) scores `0.10`; **duplicating** it (`F = 2S`, `E = S`)
scores `0.20`. The minimum size of a 1-fault-tolerant sorting network is **not
known**, so there is real headroom above any reference construction.

## The trap
The obvious recipe — place every comparator **twice** — is valid at `2S`
comparators (deleting one copy leaves its twin). It pays one full redundant
comparator per comparator and shares nothing. But fault tolerance is a **covering**
problem over deletions, not a duplication problem: after a compact sort, a *single*
failure leaves the wires only mildly disordered, so a small **shared** set of
"mop-up" comparators appended at the end can repair **many different failure cases
at once**. Comparators added after the array is already sorted are no-ops, so
deleting one of them is automatically harmless. Overlapping fault coverage this way
drives `F` well below `2S`.

## Constraints
- `3 ≤ n ≤ 12`; comparators must satisfy `0 ≤ i < j < n`; `1 ≤ F ≤ 12S`.
- Time limit 5 s, memory 512 MB. Scoring is exact and deterministic.

## Example (illustrative)
For `n = 4`, `S = 5`. Triplicating gives `F = 15`, `ratio = 0.2·5/10 = 0.10`.
Duplicating gives `F = 10`, `ratio = 0.2·5/5 = 0.20`. A compact sort plus two
shared mop-up rounds reaches `F = 8` (`E = 3`), `ratio = 0.2·5/3 ≈ 0.333` — the
same three-comparator mop-up covers every single-deletion fault.
