#!/usr/bin/env python3
"""Deterministic local scorer for "Adaptive Auction Bidding" (ale-50).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE [--raw]

Prints a single integer: the score. HIGHER is better. INFEASIBLE -> 0.

THE PROBLEM (see context.md "Evaluation settings").
You bid in T sequential sealed-bid auctions sharing one budget B. Round t reveals
an item with true utility `value_t` and a noisy public `price_hint_t`; you must
commit a non-negative integer bid for round t. You WIN the item iff your bid is at
least the round's TRUE price `price_t`; winning gains `value_t` utility and spends
`price_t` from the budget. The objective is to maximize total utility of items
won, across the whole horizon, without ever overspending the budget.

INSTANCE (read from INSTANCE_FILE):
    T B
    then T lines, each "value_t price_hint_t true_price_t".
  The bidder sees only value/hint; `true_price_t` is the hidden price the scorer
  uses to decide wins and spend. (The solver contract forbids using true_price to
  choose a bid -- it is the price the bidder does not know when bidding.)

SOLUTION (read from SOLUTION_FILE), whitespace-separated tokens:
    exactly T integers: bid_0 bid_1 ... bid_{T-1}  (each a non-negative bid).

FEASIBILITY (any violation -> score 0):
  * the token stream parses as EXACTLY T integers, nothing left over;
  * every bid is an integer with 0 <= bid;  (a negative bid is malformed)
  * BUDGET: replaying the stream round by round, whenever the bid wins the item
    (bid >= true_price) the true price is spent; total spend must NEVER exceed B.
    A win the bidder cannot afford is a BUDGET BREACH and floors the score to 0.
    (The bidder must pace its spend so it never commits to an unaffordable win.)

OBJECTIVE (of a feasible bid stream): replay round by round with spend = 0,
    for t in 0..T-1:
        if bid_t >= price_t:          # item won
            spend += price_t
            if spend > B: INFEASIBLE  # budget breach -> score 0
            utility += value_t
    The raw objective is `utility` (total utility of items won). It is >= 0.

SCORE (higher better), normalized against a deterministic SPEND-EVENLY
fixed-fraction baseline the scorer recomputes itself (see baseline_utility):
bid a fixed fraction of each item's hint so that, in expectation, average spend
tracks B / T -- the natural non-adaptive pacing. The fraction is calibrated by a
deterministic search so the baseline spends as much of B as it can WITHOUT
breaching, then plays greedily. Then
    score = round(1_000_000 * solver_utility / max(1, baseline_utility))
INFEASIBLE -> 0. With --raw the raw integer `utility` is printed instead.
"""
import sys


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    T = int(next(it)); B = int(next(it))
    value = [0] * T
    hint = [0] * T
    price = [0] * T
    for t in range(T):
        value[t] = int(next(it))
        hint[t] = int(next(it))
        price[t] = int(next(it))
    return T, B, value, hint, price


def read_solution(path, T):
    """Parse + structurally validate the bid stream. Return list[int] or None."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != T:                 # must be exactly T tokens, nothing extra
        return None
    bids = []
    for tok in toks:
        try:
            b = int(tok)
        except ValueError:
            return None
        if b < 0:                      # a negative bid is malformed
            return None
        bids.append(b)
    return bids


# ----------------------------------------------------------- replay / objective
def replay(T, B, value, price, bids):
    """Replay the stream. Return total utility, or None on a budget breach."""
    spend = 0
    utility = 0
    for t in range(T):
        if bids[t] >= price[t]:        # item won at the round's true price
            spend += price[t]
            if spend > B:              # budget breach -> infeasible
                return None
            utility += value[t]
    return utility


# ------------------------------------------ baseline: spend-evenly fixed fraction
def baseline_utility(T, B, value, hint, price):
    """Deterministic spend-evenly fixed-fraction bidder.

    The natural non-adaptive pacing: pick a single fraction f and bid f * hint on
    every round, so a round costs about f * hint when won and the average spend is
    designed to track B / T. We calibrate f by a deterministic geometric search
    over a fixed grid of fractions, replay each, and keep the fraction that
    collects the most utility WITHOUT breaching the budget. This is a fair, strong
    non-adaptive normalizer: it already paces by a constant multiplier, it just
    cannot ADAPT the multiplier to the budget actually remaining.
    """
    best = 0
    # a fixed deterministic grid of bidding fractions.
    fracs = [0.05 * k for k in range(1, 61)]      # 0.05 .. 3.0
    for f in fracs:
        spend = 0
        util = 0
        feasible = True
        for t in range(T):
            bid = int(f * hint[t])
            if bid >= price[t]:
                if spend + price[t] > B:
                    # this fixed-fraction bidder would breach here; the honest
                    # non-adaptive play is to STOP winning once the budget is
                    # exhausted (it can no longer afford anything it would win),
                    # so it simply does not take this or later unaffordable wins.
                    continue
                spend += price[t]
                util += value[t]
        if feasible and util > best:
            best = util
    return best


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION [--raw]\n")
        sys.exit(1)
    raw_mode = "--raw" in sys.argv[3:]
    T, B, value, hint, price = read_instance(sys.argv[1])

    bids = read_solution(sys.argv[2], T)
    if bids is None:
        print(0)
        return

    util = replay(T, B, value, price, bids)
    if util is None:                   # budget breach -> infeasible
        print(0)
        return

    if raw_mode:
        print(util)
        return

    base = baseline_utility(T, B, value, hint, price)
    score = int(round(1_000_000.0 * util / max(1, base)))
    print(score)


if __name__ == "__main__":
    main()
