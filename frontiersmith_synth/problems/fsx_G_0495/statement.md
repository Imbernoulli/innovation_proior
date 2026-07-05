# Airwaves: A Combinatorial Spectrum Auction

A national regulator is selling a block of spectrum **licences** in a single-round,
sealed-bid **combinatorial auction**. A licence is one atomic slice of the airwaves
(think a `(region, band)` tile). Carriers value bundles **super-additively**: a
contiguous multi-region footprint is worth strictly more than the sum of its individual
regions (coverage synergy). So each carrier submits several **package bids** under
**XOR** semantics: it names several bundles-with-prices, and **at most one** of its
packages may be awarded.

You must design the **allocation + pricing rule** the regulator runs. Given all the
package bids, choose a feasible set of winning packages and post a price to each winner,
so as to **maximize total welfare** = the sum of the (declared) values of the winning
packages. This is the NP-hard **Winner-Determination Problem** of a combinatorial
auction.

## You write a program (stdin -> stdout)

Your program reads ONE JSON object (the public instance) from **stdin** and writes ONE
JSON object (your allocation + prices) to **stdout**. It is run in an isolated sandbox
and sees only the public instance below.

### Input (stdin) — the public instance
```json
{
  "name": "auction2201",
  "m": 12,                        // number of licences, indexed 0..m-1
  "n_bidders": 5,                 // carriers, indexed 0..n_bidders-1
  "bids": [                       // flat list of package bids (XOR grouped by "bidder")
    {"bidder": 0, "items": [0, 3, 7], "value": 41},
    {"bidder": 0, "items": [3],       "value": 9},
    {"bidder": 1, "items": [2, 3],    "value": 18}
    // ...
  ]
}
```
- `bids[j]` is the `j`-th package bid: carrier `bidder`, the licence set `items`
  (distinct licence indices in `0..m-1`), and its declared `value` (a positive integer).
- A carrier owns several bids; **XOR** means at most one of a carrier's bids can win.

### Output (stdout) — your allocation + prices
```json
{"win": [0, 2], "prices": [41, 18]}
```
- `win`: the indices (into `bids`) of the packages you award. Must be **distinct**.
- `prices`: same length as `win`; `prices[k]` is the price charged to the winner of
  `bids[win[k]]`.

## Feasibility (any violation scores the instance 0.0)

A plan is valid iff **all** hold:
- `win` is a list of distinct valid indices into `bids`;
- **single assignment**: no licence appears in two winning packages;
- **XOR**: no carrier owns two winning packages;
- `prices` is a list of the same length as `win`, each price **finite** with
  `0 <= prices[k] <= value(bids[win[k]])` (**individual rationality** — a winner never
  pays more than it bid).

A crash, timeout, wrong shape, non-finite number, or non-JSON output also scores 0.0.
Prices do **not** affect the welfare score; they are a feasibility requirement that
keeps the mechanism individually rational.

## Objective & scoring (deterministic; maximize)

Welfare of your plan is `w_cand = sum of value(bids[j])` over `j in win`. The evaluator
computes two references on each instance and normalizes with an affine anchor:

- `w_base` — welfare of a weak **arrival-order greedy** allocator (accept packages in
  submission order, never reordering) → anchors **0.10**.
- `w_ref`  — welfare of a strong internal solver (multi-key greedy + local search) →
  anchors **0.80**.

```
r = clamp( 0.10 + 0.70 * (w_cand - w_base) / max(1e-9, w_ref - w_base),  0, 1 )
```

Reproducing the weak greedy scores ~0.10; matching the strong internal solver scores
~0.80; **beating** it (the WDP is NP-hard and the internal solver is not optimal) pushes
toward 1.0. Your score is the mean of `r` over a fixed battery of 12 seeded instances
(including several larger, more contended held-out auctions). Everything is deterministic
and seeded — no wall-time, no randomness in scoring.

## Strategy ladder (increasing quality)

1. **Arrival-order greedy** — accept packages in submission order, skipping collisions.
   Matches `w_base` (~0.10).
2. **Value / density greedy** — reorder by value (or value-per-licence) so efficient
   bundles beat one huge blocking bid; a large jump over arrival order.
3. **Multi-key greedy + local search** — seed from several orderings, then run
   1-for-many swaps that drop the winners a package collides with whenever welfare rises.
   Approaches ~0.80.
4. **Toward-optimal WDP** — branch-and-bound / ILP / stronger metaheuristics to pass the
   internal reference toward 1.0.
