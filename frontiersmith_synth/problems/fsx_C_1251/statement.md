# Which Custom Instructions Are Worth The Silicon

## Problem

You are designing the custom-instruction extension for an accelerator. A vendor library
gives you `C` **candidate instructions** (each is a small fused subgraph, e.g. a
multiply-add or a shift-and-mask). Candidate `c` has an **area cost** `area[c]`
(silicon it burns if implemented), a **base width** `size[c]` (the number of ordinary
1-cycle ops it fuses together) and a **fused cost** `cost[c] < size[c]` (cycles the
fused instruction itself takes). Implementing `c` and matching it at one spot in the
code turns `size[c]` cycles of base execution into `cost[c]` cycles, banking
`size[c]-cost[c]` cycles of savings *at that spot*.

You are told, for one or more **application code streams**, every position where every
candidate *could* match (a "subgraph occurrence": candidate `c` matching at position
`start` would replace positions `start..start+size[c]-1`). Different candidates can have
occurrences that cover the exact same positions — they are alternative fusions of the
same code region, and at most one of them can ever actually apply there.

Your job: choose which candidates to actually burn into silicon. Two hard budgets limit
you: an **encoding-space budget** `K` (only `K` opcodes fit in the ISA, i.e. at most `K`
candidates total) and an **area budget** `A` (`sum(area[c]) <= A` over chosen `c`).

## Input (stdin)
```
K A
C
area[0] size[0] cost[0]
...
area[C-1] size[C-1] cost[C-1]
M
L_0 O_0
c start        (O_0 lines: candidate c has an occurrence at position start in app 0)
...
L_1 O_1
...
```
`M` application streams follow; app `m` has length `L_m` (positions `0..L_m-1`) and
`O_m` listed occurrences. All ids/positions are consistent (generator-guaranteed).

## Output (stdout)
```
S
id_1 id_2 ... id_S
```
`S` distinct candidate ids (0-indexed, `0 <= S`), in any order.

## Feasibility
`S <= K`, `sum(area[id])<= A` over the chosen ids, ids distinct and in `[0,C-1]`. Any
parse failure, out-of-range id, duplicate, non-finite token, or budget violation scores
`Ratio: 0.0`.

## Objective (maximize cycles saved)
Given your chosen set `S`, the checker recompiles every application with a single fixed,
deterministic pass: among occurrences whose candidate is in `S`, visit them in order of
(start position ascending; then savings `size[c]-cost[c]` descending; then size
descending; then candidate id ascending). An occurrence is applied iff every position it
covers is still unclaimed by an earlier occurrence in this order; applying it claims
those positions and banks `size[c]-cost[c]` cycles. `F` = total cycles banked across all
applications. This rule is fixed by the problem — you only control which candidates are
*available* to it, not how conflicts are broken.

## Scoring
The checker also runs a naive construction — candidates taken in raw id order,
first-fit under both `K` and `A` — through the same recompilation pass to get a baseline
`B`. `Ratio = min(1000, 100*F/max(1e-9,B)) / 1000`.

## Constraints
`1 <= C <= 60`, `1 <= K < C`, `1 <= M <= 3`, each `L_m <= 800`, total occurrences across
all apps `<= 2000`, `1 <= cost[c] < size[c] <= 8`, `1 <= area[c] <= 7`. Time limit 5s.

## Example (worked score)
One app, `L=6`. Candidates: `0:(area=2,size=4,cost=1)`, `1:(area=2,size=4,cost=3)` both
occur at `start=0` (same 4-wide region — alternative fusions of it); candidate
`2:(area=1,size=2,cost=1)` occurs at `start=4` (disjoint tail). `K=2, A=5`.

Baseline (id order, first-fit): picks `{0,1}` (area `2+2=4<=5`, count `2<=2`). Recompile:
at `start=0` both occurrences are eligible; candidate `0` has the larger savings
(`4-1=3` vs `4-3=1`) so it wins by the tie rule, banking `3`; candidate `1`'s occurrence
finds its positions already claimed, banks `0`. `B = 3`.

A submission choosing `{0,2}` instead (drop the redundant alternative, spend the freed
slot on the disjoint tail candidate): banks `3` at `start=0` plus `2-1=1` at `start=4`,
`F = 4`. `Ratio = min(1000, 100*4/3)/1000 = 0.1333`. Choosing `{0,1}` like the baseline
caps out at `0.1`; choosing `{1,2}` instead (the strictly worse fusion of the same spot,
plus the tail) only reaches `1+1=2`, scoring below baseline.
