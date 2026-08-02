# Variation-Aware Clock Tree: Buffer Insertion Under a Skew and Power Budget

## Problem
A clock source drives `K` independent sinks over `K` routed nets (a star
clock tree). Net `i` has a fixed wire delay `w[i]`; nets differ in wire
delay because they are routed to sinks at different distances.

You may insert buffers from a fixed library of `m` types. Type `t` has a
nominal per-buffer delay `D[t]`, a per-buffer power cost `P[t]`, and a
per-buffer delay offset `Var[t][c]` for each of `C` **deterministic process
corners** (fixed numbers baked into every test case -- these are worst-case
manufacturing corners, not randomness). Under corner `c`, a buffer of type
`t` contributes delay `D[t] + Var[t][c]` instead of `D[t]`. Buffers only add
delay, they never subtract it.

For net `i` you choose a non-negative integer count `count[i][t]` of buffers
of each type `t`; the net's nominal arrival delay is
`w[i] + sum_t count[i][t]*D[t]`, and its arrival delay under corner `c` is
`w[i] + sum_t count[i][t]*(D[t]+Var[t][c])`.

## Input (stdin)
```
K m C
D[0] P[0] Var[0][0] ... Var[0][C-1]
...
D[m-1] P[m-1] Var[m-1][0] ... Var[m-1][C-1]
NomSkewBudget WorstSkewBudget
w[0]
...
w[K-1]
```
All values are non-negative integers.

## Output (stdout)
Exactly `K` lines, each with `m` non-negative integers:
```
count[i][0] ... count[i][m-1]
```

## Feasibility
1. Exactly `K*m` valid non-negative integer tokens (no decimals, no
   scientific notation, no `nan`/`inf`) -- any other token count or malformed
   token scores 0.
2. **Nominal skew**: let `nom[i] = w[i] + sum_t count[i][t]*D[t]`. Then
   `max(nom) - min(nom) <= NomSkewBudget`.
3. **Worst-case (process-corner) skew**: for every corner `c`, let
   `dc[i] = w[i] + sum_t count[i][t]*(D[t]+Var[t][c])`. The worst-case skew
   `max_c (max(dc) - min(dc))` must be `<= WorstSkewBudget`.

Any violation of 1-3 scores `Ratio: 0.0`. `NomSkewBudget` is always strictly
less than the raw wire-delay spread `max(w) - min(w)`, so `count` cannot be
all zeros.

## Objective
Minimize total power `F = sum_i sum_t count[i][t]*P[t]`.

## Scoring
The checker builds its own reference plan `B`: buffer type 0 (the low-
variation "safe" type) only, adding buffers to every net until its nominal
delay reaches `max(w)`. This plan is always feasible but power-wasteful.
With your feasible total power `F`:
```
Ratio = min(1, 0.1 * B / F)
```
Fewer/cheaper buffers score higher, but only among plans that stay within
**both** skew budgets.

## Why this isn't just "balance the delays"
`NomSkewBudget` is generous -- it lets you leave a real fraction of the raw
wire-delay spread unbalanced, so you never need to fully equalize nominal
delay. `WorstSkewBudget` sits only a little above it. Every inserted
buffer's corner offset `Var[t][c]` adds to whichever net received it; a net
that got many buffers chasing a smaller nominal gap becomes far more exposed
to the process corners than one left alone. Racing to close the *whole*
nominal gap with the buffer type that needs the fewest insertions looks
efficient in isolation, yet it concentrates variation exposure on exactly
the nets that received the most "help," pushing the worst-case skew over
budget. Closing only the part of the gap the budget actually requires, with
a low-variation type, can be both cheaper and safer than a fully balanced
tree.

## Example (worked score, illustrative shape only)
Suppose `K=2`, wire delays `10` and `40` (gap 30), and a plan uses `5`
buffers of type 0 (`P=5`) on the first net and none on the second, giving
`F = 25`. If the checker's own reference plan needed `10` type-0 buffers
(`B = 50`) to reach feasibility, `Ratio = min(1, 0.1*50/25) = 0.2` --
provided the plan also satisfies both skew budgets.

## Constraints
`6 <= K <= 15`, `m = 3`, `C = 4`, all delays/powers/variations fit in a
32-bit signed integer, budgets are non-negative integers. Time limit 5s,
memory 512MB.
