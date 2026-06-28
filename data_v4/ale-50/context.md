# Adaptive Auction Bidding (online resource buy)

## Research question

A buyer takes part in a sequence of `T` sealed-bid auctions, one per round, sharing a single fixed
**budget** `B`. In round `t` an item appears with a true **utility** `value_t` (what the buyer gains
by winning it) and a public but **noisy** `price_hint_t` (a signal of how expensive the item will be).
Before the round's hidden **true price** `price_t` is revealed — and before any later round is seen —
the buyer must commit a non-negative integer **bid** for round `t`. The buyer **wins** the item iff
the bid is at least the true price; on a win the buyer gains `value_t` utility and spends `price_t`
out of the budget.

The objective is to **maximize the total utility of items won across the whole horizon** subject to
one hard rule: the cumulative spend must never exceed `B`. Because the budget affords only a minority
of the rounds, and the high-value items are heavy-tailed and arrive in an unknown order, this is an
online sequential decision problem with no exact answer: it is judged by a continuous score. The only
lever is the heuristic that decides **how much to bid each round** given everything seen so far.

## Input / output contract

This is an interactive/online problem cast in the offline ALE-Bench single-file form: the whole round
stream is present in the instance, but the *contract* (enforced by the scorer) is that round `t`'s bid
may depend only on the values/hints of rounds `0..t` and the budget already spent — never on a future
round and never on the realized true price of any round.

- **Input (stdin).** The first line is `T B` with the number of rounds `300 <= T <= 1200` and the
  integer budget `B >= 1`. Then follow `T` lines, the `t`-th being three integers
  `value_t price_hint_t true_price_t`:
  - `value_t` — the item's true utility (heavy-tailed; most items are modest, a few are worth far
    more);
  - `price_hint_t` — the **public noisy** price signal the bidder sees;
  - `true_price_t` — the **hidden** true price. It is in the instance so the scorer can decide wins
    and spend, but it is the number the bidder does **not** know when bidding. The bidder reads it only
    *after* its round-`t` bid is fixed, exactly as a real judge would announce the outcome.
- **Output (stdout).** Exactly `T` whitespace-separated non-negative integers `bid_0 ... bid_{T-1}` —
  the bid committed in each round, in order. Nothing may follow the last bid.
- **Time limit:** about 2 seconds. **Memory:** 256 MB.

Example shape: with `T = 500`, a valid output is 500 integers. Bidding `0` on every round (winning
nothing, spending nothing) is always feasible and scores `0`.

## Background

This is the classic **budget-constrained repeated-auction / online-knapsack** problem. You want the
items with the best utility-per-cost, but you cannot see the future, so you cannot sort and pick — and
every win you take eats a shared budget that the high-value items arriving later also need. Two
reference points frame the design.

- **Spend-evenly fixed-fraction bidding (the baseline).** Pick a single fraction `f` and bid
  `f * price_hint_t` on every round, so the average spend is designed to track `B / T`. Calibrate `f`
  to the largest value that does not exhaust the budget too early. This is the natural **non-adaptive**
  pacing: it paces by a constant multiplier. Its weakness is that the multiplier is **fixed** — it
  cannot react when the realized prices run hot (draining the budget faster than planned, forcing it
  to stop early and miss late high-value items) or cold (leaving budget unspent, wasted utility). It
  spends the same way regardless of how much budget is actually left. This is the strategy the scorer
  normalizes against.

- **Dual-variable budget pacing (the strong method).** The established strong approach for budgeted
  online allocation is **primal-dual pacing**: maintain a dual price `lambda >= 0` on the budget
  constraint — the *shadow cost* of spending one unit of budget — and accept an item iff its utility
  beats the dual cost of the budget it would consume, `value_t >= lambda * (expected price)`. After
  each round, update `lambda` online (online mirror descent / an exponentiated-gradient step) against
  the per-round budget target `B / T`: spend faster than the target and `lambda` rises (become
  stingier); spend slower and `lambda` falls (become more willing). The single dual variable turns a
  global budget constraint into a local per-round bid decision that **automatically tracks** `B / T`,
  reallocating the budget toward the genuinely high utility-per-cost items as the realized prices
  reveal themselves. The open design choices are the step size, how to estimate an item's expected
  cost from a noisy hint, and how to bake in a hard guarantee that the budget is **never** breached
  (a single overspend floors the score to 0).

## Evaluation settings

For a fixed seed the generator (`verify/gen.py`) produces one instance. A solver's output is scored
exactly as `verify/score.py` computes:

- **Feasibility / floor (any violation -> score 0).** The token stream must parse as **exactly** `T`
  integers, with nothing left over; every bid must be a non-negative integer. Replaying the stream
  round by round, whenever a bid wins its item (`bid_t >= true_price_t`) the true price is spent;
  the cumulative spend must **never exceed `B`**. A win the bidder cannot afford is a **budget breach**
  and floors the score to `0`.

- **Objective (of a feasible bid stream).** Replay with `spend = 0`; for each round, if
  `bid_t >= true_price_t` then `spend += true_price_t` (breach if `spend > B`) and
  `utility += value_t`. The raw objective is the total `utility` of items won (always `>= 0`).

- **Normalized score.** The scorer recomputes a deterministic **spend-evenly fixed-fraction** baseline:
  over a fixed grid of fractions `f`, bid `int(f * price_hint_t)` each round, skip a win it cannot
  afford, and keep the fraction collecting the most utility without breaching. Then
  `score = round(1_000_000 * solver_utility / max(1, baseline_utility))`.
  The baseline scores about `1_000_000`; an adaptive pacing scheme that allocates the budget to better
  items scores higher; an infeasible output scores `0`.

**How instances are generated** (`verify/gen.py`, parameter = integer seed). `T` is drawn in
`[300, 1200]`. Each item's `value` is heavy-tailed (a mixture: ~80% small, ~17% medium, ~3% very
large), so *which* items you save the budget for matters. A per-item base price is correlated with the
value (correlation ~0.4..0.8) plus independent mass and item-level spread; the **public hint** is the
base price observed with multiplicative noise in `[0.80, 1.20]`, and the **hidden true price** is the
base times an *unseen* per-round multiplier in `[0.70, 1.40]` — so a bid must hedge against price
uncertainty. The budget `B` is a fraction `[0.20, 0.45]` of the total true price of all items, so only
a minority of rounds is affordable: pacing is forced, which is exactly the regime where an adaptive
dual-variable schedule beats a fixed-fraction sweep.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes a feasible
solution (exactly `T` non-negative integer bids) to stdout. The scaffold below already emits a valid
(all-zero, score-0) bid stream; the method replaces the TODO with the dual-variable pacing loop and
its mirror-descent update, while preserving the never-breach cap.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long T, B;
    if (!(cin >> T >> B)) return 0;
    vector<long long> value(T), hint(T), price(T);
    for (long long t = 0; t < T; t++)
        cin >> value[t] >> hint[t] >> price[t];   // 3rd col is the hidden price

    // A feasible fallback: bid 0 on every round (win nothing, spend nothing).
    // Always valid -- never breaches the budget, never malformed (scores 0).
    vector<long long> bids(T, 0);

    // TODO heuristic: maintain a budget-pacing dual variable lambda (the shadow
    // cost of budget). For each round, accept the item iff value beats the dual
    // cost of its expected price; bid a calibrated fraction of the hint CAPPED at
    // the remaining budget (so a win can never breach); then update lambda by an
    // online mirror-descent step toward the per-round budget target B / T.

    // emit exactly T bids
    string out;
    for (long long t = 0; t < T; t++) {
        out += to_string(bids[t]);
        out += (t + 1 == T) ? '\n' : ' ';
    }
    cout << out;
    return 0;
}
```
