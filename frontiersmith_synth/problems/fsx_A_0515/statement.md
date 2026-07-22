# The Clearance Ladder: Screening Patient Buyers with a Price Spike

## Problem
You run a boutique clearance sale over `T = 30` days. You set an **integer price**
`p_t` for every day `t = 1..T` in advance. Then `N` forward-looking customers arrive
and each decides, on their own, the single best day to buy one unit.

Customer `i` is described by four numbers:
- `v` — the value they place on the item;
- `a` — the day they enter the market (they cannot buy before day `a`);
- `h` — their deadline (they cannot buy after day `h`);
- `D` — their patience, a discount **permille**, so their per-day discount is
  `d = D / 1000`.

The store has a total of `K` units in stock, and can hand over at most `s` units
on any single day (shelf capacity).

## Input (stdin)
```
T N K s PMAX p0
v_1 a_1 h_1 D_1
...
v_N a_N h_N D_N
```
All values are integers. `p0` is a reference price (see Scoring). Customers are
listed in the exact order they will be **processed**.

## Output (stdout)
Exactly `T` integers `p_1 ... p_T` (whitespace-separated), each in `[0, PMAX]`.

## Feasibility
The output must be exactly `T` integers, each finite and in `[0, PMAX]`. Any other
output (wrong count, non-integer, out of range, non-finite) scores `Ratio: 0.0`.

## Buyer best response (how revenue is produced)
Customers are processed in input order. Maintain remaining global stock and a
per-day sold count. For the current customer `(v, a, h, D)` with `d = D/1000`:
- if global stock is exhausted, they buy nothing;
- otherwise consider every day `t` with `a <= t <= h` whose day-`t` sold count is
  still below `s`. Their **surplus** on day `t` is `d^t * (v - p_t)`;
- let `t*` be the day of **maximum surplus** (ties → the earliest such day). If that
  maximum surplus is strictly positive, the customer buys one unit on day `t*`,
  pays `p_{t*}`, consuming one unit of stock and one unit of day-`t*` capacity.
  Otherwise they buy nothing.

Your revenue `F` is the total of all prices paid.

## Objective
**Maximize revenue `F`.** Note the twist: your price path is transformed by the
customers' best response *before* it is scored, so you must optimize the induced
buying behavior, not the raw schedule. A patient customer will wait for the
discount-cheapest day inside their window; an impatient one (small `d`, short
window) must buy early even at a high price; a customer only in the market
mid-run can never reach an early or late bargain.

## Scoring
Let `B` be the revenue of the **constant schedule** `p_t = p0` for all `t`, replayed
by the same best-response rule (the store's do-nothing baseline). With maximization
normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the constant `p0` schedule scores `Ratio = 0.1`; ten times its revenue
caps at `1.0`.

## Constraints
- `T = 30`, `0 <= p_t <= PMAX` (`PMAX = 120`), `1 <= N <= 120`.
- `1 <= a <= h <= T`, `1 <= K <= N`, `1 <= s <= N`, `500 <= D <= 1000`.
- Time limit 5s, memory 512m.

## Example
With three customer clusters — an impatient low-value early crowd, a high-value
crowd present only around days 8–16, and a patient low-value bargain crowd — a
monotone markdown (start high, mark down to a floor) lets the patient crowd wait
for the floor and forces the mid crowd to the same low prices. A **non-monotone**
path (cheap early, a mid-window price *spike*, then a late markdown) charges each
cluster its own price and earns strictly more.
