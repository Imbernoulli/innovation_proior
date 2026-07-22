# Estate Auction House: Two Hundred Lots

## Problem
An estate auction house is liquidating `n` lots to `m` registered bidders. Bidder `j`
has a **hard budget** `b_j` (cash that never replenishes) and a **value** `v_{i,j} >= 0`
for lot `i` (`0` means bidder `j` has no interest in lot `i`). All values and budgets
are given to you in advance.

You design the entire sale: the **order** in which the `n` lots go under the hammer,
and a **reserve price** for every lot. The house then runs the sale mechanically, lot
by lot, in your chosen order:

- For the current lot `i` with your chosen reserve `r`, every bidder `j` with
  `v_{i,j} > 0` submits a bid `bid_j = min(v_{i,j}, remaining_j)`, where `remaining_j`
  is bidder `j`'s budget **left after all earlier lots**. A bidder only actually
  participates if `bid_j >= r`.
- If nobody participates, the lot goes unsold (contributes `0` revenue, no budgets
  change).
- If exactly one bidder participates, they win and pay the reserve `r`.
- If two or more participate, the highest bidder wins (ties broken by the smaller
  bidder index) and pays `max(r, second-highest bid)` — a standard second-price
  auction with reserve.
- The winner's `remaining_j` is reduced by the price paid; every other bidder's
  budget is untouched.

Your **total revenue** is the sum of the clearing prices over all `n` lots. Because a
bidder's remaining budget only shrinks when *they* win something, the order you pick
governs a whole **budget-depletion trajectory**: whichever bidder wins early lots has
less firepower for later lots they also wanted, and setting a reserve too high can
lose a sale entirely while setting it too low leaves money on the table. A bidder who
has deep pockets and broad taste often collides with several other bidders across many
lots (a "bidder-overlap" pattern implicit in the value table) — burning that bidder's
budget too fast in one place starves the competition it would otherwise provide
elsewhere, and stalling it too long lets its rivals' *own* budgets quietly run dry
first. Good schedules typically **interleave** which lots draw on which bidders rather
than sorting purely by sticker value.

## Input (stdin)
```
n m
v_1_1 v_1_2 ... v_1_m
...
v_n_1 v_n_2 ... v_n_m
b_1 b_2 ... b_m
```
`v_i_j` and `b_j` are non-negative integers. Lots and bidders are 1-indexed.

## Output (stdout)
Exactly `n` lines, each `item_id reserve`, listing lots in the exact order they are
auctioned. `item_id` (1..n) must form a permutation of every lot. `reserve` is an
integer with `0 <= reserve <= 5000000`.

## Feasibility
- Exactly `2n` integer tokens total, i.e. `n` `(item_id, reserve)` pairs.
- `item_id` values are a permutation of `1..n` (each lot exactly once).
- Every `reserve` is an integer in `[0, 5000000]`.
- All tokens finite integers (no `nan`/`inf`, no malformed tokens).

Any violation scores `Ratio: 0.0`.

## Objective
Maximize total revenue `F` = sum of clearing prices over the sale, replayed exactly
as described above in your chosen order and reserves.

## Scoring
The checker also runs the **same replay mechanics** using the trivial schedule
(lots in input order `1..n`, reserve `0` everywhere) to get an internal baseline `B`.
Score:
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
Matching the trivial schedule scores `0.1`; beating it `10x` caps the score at `1.0`.

## Constraints
- `6 <= n <= 200`, `4 <= m <= 46` across the test ladder.
- Every lot has at least one bidder with positive value (so `B > 0` always).
- Deterministic exact-integer scoring; no randomness or timing in the score.

## Example
`n=3, m=2`, values `[[100,10],[10,100],[60,60]]`, budgets `[80,80]`.
Trivial schedule (order `1,2,3`, reserve `0`): lot 1 -> bidder 1 wins (bid 80 capped
by budget) paying `10`; lot 2 -> bidder 2 wins paying `10`; lot 3 -> both bid `60`,
price `60`, one wins (budget now `70-60=10` say bidder1 wins, budget left `70-60=10`,
here budgets already at 70 each). Total `B` is whatever that replay yields; a smarter
order/reserve choice on lot 3 (raising its reserve, or selling it before the budgets
are spent) captures more of that `60` head-to-head collision instead of losing it to a
starved bidder — that is the kind of gain `F` should chase. (Illustrative only — exact
numbers depend on the real replay.)
