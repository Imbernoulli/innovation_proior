# Secret Districts: Maximum-Reach Courier Routes

## Problem
A courier company operates over `n` addresses in a city. There are `m` basic **moves**; each move
is a permutation of the `n` addresses (a re-routing rule). A **route** (word) is a sequence of at
most `L` moves; you may use a move forwards (token `j`, apply move `j`) or in reverse (token `-j`,
apply move `j` inverse). Applying route `w` to an address `p` yields another address `w(p)`.

The city is secretly divided into equal-size **districts** (an invariant partition of the addresses:
every move maps whole districts onto whole districts). The partition is **not given** — you only see
the moves as raw permutations of the `n` addresses. You start with a set `S` of `t` parcels, all in
one district, and a budget of `k` routes. Choosing routes `w_1,...,w_k`, you **cover** the set of
addresses reachable as images of your parcels:
```
covered = union over chosen routes w of { w(p) : p in S }.
```

## Input (stdin)
```
n m k L t
<move 1: n integers, a permutation of 0..n-1>
...
<move m: n integers>
<S: t distinct integers in 0..n-1>
```
Move `j` sends address `x` to the `x`-th listed integer. Exactly one move changes which district an
address lies in; the others only shuffle addresses within their own district.

## Output (stdout)
Up to `k` lines, one route per line: space-separated tokens, each in `-m..-1` or `1..m` (token `0`
means "no move", so a line `0` is the empty/identity route). A route must have at most `L` tokens.

## Feasibility
More than `k` routes, a route longer than `L`, or any token outside `-m..m` scores `Ratio: 0.0`.

## Objective (maximize)
`F = |covered|`, the number of distinct addresses your `k` routes reach from `S`.

## Scoring
Let `B` be the coverage of the reference construction that spams move `1` at lengths
`0,1,2,...` (capped at `L`) over `k` routes — it only re-shuffles the seed district. Then
```
Ratio = min(1000, 100 * F / B) / 1000
```
so the reference construction scores about `0.1` and reaching ten times its coverage caps at `1.0`.
The exact districts, their size, and the within-district twist of each move live only in the input —
you must read the permutations and exploit their structure.

## Constraints
`n <= ~1000`, `m = 6`, time limit 5s, memory 512MB.

## Example (illustrative)
With two districts `{0,1}` and `{2,3}`, moves `move1 = (0 1)(2 3)` (intra-district) and
`move2 = (0 2)(1 3)` (swaps the districts), `S = {0}`, `k = 2`, `L = 1`: routes `0` and `2` cover
`{0, 2}` — one address in each district — beating any two intra-district shuffles that stay in
`{0,1}`. Real instances hide which move is the district-swapper among six look-alike permutations.
