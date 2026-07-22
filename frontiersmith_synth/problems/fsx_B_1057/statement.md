# The Pawnbroker's One Shelf

## Problem
A pawn shop has exactly one display shelf of capacity `C` (grams). Over the course
of a day, `N` customers arrive in a fixed, known order, each asking to inspect one
item from the shop's catalogue of `K` distinct items (item `i` weighs `size_i`
grams). You are the shopkeeper AND a clairvoyant appraiser: you see the entire
day's schedule of visits in advance and must decide, once, a full plan for the day.

If the requested item is already on the shelf, the customer inspects it for free
(a **hit**) and nothing changes. If it is not on the shelf (a **miss**), you must
fetch it from the back vault; this always costs `size_i` grams of handling effort,
regardless of what you do next. Having fetched it, you then choose:

- **Bypass** — hand it to the customer and put it straight back in the vault
  without ever placing it on the shelf, or
- **Admit** — put it on the shelf, evicting zero or more currently-shelved items
  (by name) to make room, provided the shelf's total weight after the swap does
  not exceed `C`.

Admission is not automatic: unlike a plain eviction policy, "fetch it but never
shelve it" is a first-class choice you must weigh against every alternative.
Because items have different, arbitrary sizes, deciding *which* items to evict to
fit a newcomer is itself a packing decision, not just a queue operation — this
combination (variable sizes + an explicit refuse-to-shelve action) makes finding
the truly optimal day plan intractable in general; you are scored against how
close you get.

## Input (stdin)
```
N K C
size_1 size_2 ... size_K
id_1 id_2 ... id_N
```
`id_t` (1-indexed into `1..K`) is the item the `t`-th customer asks for.

## Output (stdout)
Exactly `N` lines, the `t`-th line describing your action for visit `t`, in the
same order as the input:
- `H` — you claim item `id_t` is already on the shelf (must match reality).
- `B` — item `id_t` was a miss; you bypass it.
- `A k e_1 e_2 ... e_k` — item `id_t` was a miss; you admit it, evicting the `k`
  listed (currently-shelved) item ids first (`k` may be 0).

## Feasibility
Replayed left to right against a shelf state that starts empty and is updated only
by your `A` actions:
- If `id_t` is genuinely on the shelf at that point, your line must be exactly `H`.
- Otherwise it must be `B`, or `A k e_1..e_k` where every `e_j` is currently on the
  shelf, all `e_j` are distinct, and (shelf weight) − (sum of evicted sizes) +
  `size_{id_t}` ≤ `C`.
- Exactly `N` lines, no extra tokens, all values finite integers.
Any violation scores `Ratio: 0.0`.

## Objective
Minimize `F` = the sum of `size_{id_t}` over every visit that was a miss (fetching
from the vault costs the same whether you then bypass or admit).

## Scoring
Let `B` be the checker's own trivial construction: bypass every single visit (never
shelve anything), so `B` = the sum of `size_{id_t}` over **all** `N` visits. Then
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = max(0.0, sc) / 1000.0
```
Never shelving anything scores `Ratio ~= 0.1`; cutting fetch cost to a tenth of the
naive baseline caps the score at `1.0`.

## Constraints
- `4 <= K <= 300`, `12 <= N <= 3000`, `1 <= size_i <= 2*10^6`.
- Some items may be **larger than `C`** — such an item can never be shelved at all
  (any `A` action on it is infeasible); it must always be bypassed, and every visit
  to it is a forced miss no matter what you do.
- Time limit 5s, memory 512m.

## Example
Suppose the schedule loops tightly through a small cluster of items that together
weigh well under `C`, but is punctuated by many other items that each appear only
once. Always shelving on a miss (evicting whatever was least recently touched) lets
those one-time items repeatedly bump the loop's members off the shelf, so the loop
itself starts missing on every return visit — the shelf spends its capacity on
items that will never be asked for again. A plan that instead reads the whole
day's schedule first, recognizes which items have no further visits ahead of them,
and refuses to shelve those (bypassing them, evicting nothing) keeps the loop
intact and hitting for the rest of the day. Which items are one-time and which
recur, and by how much, is determined entirely by the input's actual sequence —
the checker computes `F` exactly as defined above from your submitted actions.
