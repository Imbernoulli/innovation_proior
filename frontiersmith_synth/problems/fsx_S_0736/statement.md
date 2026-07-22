# Stall Menu DAG

## Problem

A food market is organized as **stalls**: stall `0` is the market entrance (the root), and
every other stall represents one *kind of dish* that some recipe needs as an ingredient. Each
stall `i` offers `M_i` candidate **dishes**. Dish `k` at stall `i` has a fixed prep-cost and a
list of **ingredient-stalls** (children) — other stalls whose dish must *also* be prepared to
serve dish `k`. Every ingredient-stall listed by any dish has a strictly larger index than the
stall offering that dish, so the market has no circular dependency no matter which dishes are
picked.

You must choose exactly one dish per stall. Starting from the root's fixed dish (which lists
every stall the market actually offers as an ingredient), follow chosen dishes' ingredient
lists to determine which stalls are **reachable** — i.e., actually needed to serve everything
the root requires. A stall that is reachable through several different dishes' ingredient
lists is **prepared once and reused**: you pay its chosen dish's prep-cost a single time no
matter how many other dishes rely on it. The market's total cost is the sum of prep-costs of
every reachable stall's chosen dish, each counted exactly once.

This means the cost of choosing a dish for stall `i` is **not** a property of stall `i` alone:
it depends on which other stalls end up sharing (or not sharing) the same downstream
ingredient-stalls. A dish that looks expensive on its own can be the better pick precisely
because many other stalls can reuse the ingredient-stall it pulls in; a dish that looks cheap
on its own can be wasteful if its ingredients are needed by nobody else. Minimize the total
market cost.

## Input (stdin)

```
N R
r_1 r_2 ... r_R
M_0
cost_{0,0} L_{0,0} child_1 ... child_L
...  (M_0 lines)
M_1
...
```
`N` stalls (indices `0..N-1`), `R` roots (indices that must be served). Then, for each stall
`i = 0..N-1` in order: a line with `M_i` (its number of candidate dishes), followed by `M_i`
lines, each `cost L child_1 ... child_L` — the dish's prep-cost, its number of
ingredient-stalls, then their indices (each strictly greater than `i`).

## Output (stdout)

Exactly `N` integers, one per line (or whitespace-separated): line `i` is the index (into
stall `i`'s dish list, `0`-based) of the dish you choose for stall `i`. You must provide a
value for every stall, including ones that end up unreachable (any in-range index is fine for
those — they don't affect the score).

## Feasibility

Every printed index must be an integer in `[0, M_i)` for its stall, and you must print exactly
`N` tokens. The induced ingredient graph (restricted to reachable stalls, following only the
chosen dishes) must be acyclic — guaranteed automatically here since every ingredient index
exceeds its parent's index, but a malformed/garbage submission is still rejected.

## Objective / Scoring

Let `F` = sum of prep-costs of every reachable stall's chosen dish, each counted once (reached
via BFS/DFS from the roots along chosen ingredient links). Let `B` be the same total cost under
the checker's own baseline (always choosing dish index `0`). The score is

```
Ratio = min(1, 0.1 * B / F)
```

so cheaper `F` scores higher; an infeasible/malformed submission scores `0`.

## Example (worked, illustrative shapes only)

Stalls: `0` root -> requires stalls `1,2`. Stall `1` has one dish: cost `3`, no ingredients.
Stall `2` has two dishes: dish `0` = cost `5`, ingredients `[1]`; dish `1` = cost `2`, no
ingredients. Choosing dish `0` at stall `2` costs `5` for stall 2 *plus* stall 1's `3` — but
stall `1` is already being paid for anyway (it's required directly by the root), so reusing it
is free: total = `3 (stall 1) + 5 (stall 2) = 8`. Choosing dish `1` at stall `2` instead gives
total = `3 + 2 = 5`, cheaper here since stall 2's ingredient-free dish avoids nothing shared.
Real instances plant much larger versions of this trade-off across many stalls simultaneously,
where a shared ingredient only pays off once enough stalls jointly commit to it.

## Constraints

`N` up to a couple thousand, `M_i` in `1..3` (the root has `M_0 = 1`; leaf/kernel/private
stalls also have `M_i = 1`), costs are positive integers, each ingredient list has `0..2`
entries. Time limit 5s, memory 512MB, 10 test cases of growing size, each input file well
under 5MB.
