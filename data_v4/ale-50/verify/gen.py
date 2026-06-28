#!/usr/bin/env python3
"""Instance generator for "Adaptive Auction Bidding" (ale-50).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    T B
    value_0 price_hint_0 true_price_0
    value_1 price_hint_1 true_price_1
    ...
    value_{T-1} price_hint_{T-1} true_price_{T-1}

Meaning. You take part in a sequence of `T` sealed-bid auctions, one per round,
with a single shared budget `B`. In round `t` an item appears with a true
utility `value_t` (what you gain if you win it) and a public, NOISY `price_hint_t`
(a signal of how expensive the item will be). You must commit a non-negative
integer BID for round `t` -- a price you are willing to pay -- using ONLY the
information visible by round t: the values and hints of rounds 0..t and the
budget already spent. You win the item iff your bid is at least the round's TRUE
price; on a win you gain `value_t` utility and spend the true price.

The third column, `true_price_t`, is the round's hidden true price. It is written
to the instance for the SCORER (which replays the stream and decides wins/spend),
but it is the price the bidder does NOT know when bidding: the bidder sees only
`value` and `price_hint`. The contract (enforced conceptually like ale-42's
hidden field) is that round t's bid is a function of `value`/`hint` of rounds
<= t and the spend so far -- NEVER of any `true_price`, and never of a future
round. A solver that bid exactly each true price would be cheating the online
contract; the legitimate game is to bid against the UNCERTAIN price from the hint.

Causality + budget (enforced by score.py). The scorer replays round by round.
Winning must never make total spend exceed `B`; a budget breach floors the score
to 0. The bidder must therefore PACE its spend across the whole horizon.

The whole game is PACING: bid too high early and you win cheap-utility items,
burn the budget, and have nothing left for the high-value items that arrive
later (or you breach the budget and score 0); bid too low and you win almost
nothing. The strong heuristic maintains a budget-pacing dual variable updated
online so that average spend tracks B / T and the budget is spent on the items
with the best utility-per-cost.

Instance regime (deterministic from the seed):
  * T in [300, 1200] rounds.
  * values are heavy-tailed (a few items are worth far more than the rest), so
    WHICH items you save the budget for matters.
  * price hints are correlated with value (expensive items tend to be valuable)
    but noisy, and the hidden true price is the hint times an unseen per-round
    multiplier -- so a bid must hedge against price uncertainty.
  * the budget B is a fraction (about 0.20..0.45) of the total true price of all
    items, so you can afford only a minority of rounds: pacing is forced.
"""
import sys
import random

SCALE = 1000


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x5011_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    T = rng.randint(300, 1200)

    # heavy-tailed values: most items modest, a few very valuable.
    # base price level per item correlates with value but with its own noise; the
    # PUBLIC hint is a noisy observation of the base price level, and the HIDDEN
    # true price is base_price * (unseen per-round multiplier).
    rounds = []          # (value, price_hint, true_price)
    total_price = 0
    for _ in range(T):
        # heavy-tailed value via a mixture: mostly small, occasionally huge.
        u = rng.random()
        if u < 0.80:
            value = rng.randint(1, 200)
        elif u < 0.97:
            value = rng.randint(200, 900)
        else:
            value = rng.randint(900, SCALE * 3)

        # base price level: correlated with value (corr ~0.6) plus independent mass.
        corr = rng.uniform(0.4, 0.8)
        base = corr * value + (1.0 - corr) * rng.uniform(50, 600)
        base *= rng.uniform(0.6, 1.4)          # item-level price spread
        base = max(1.0, base)

        # public hint = base price observed with multiplicative noise.
        hint = base * rng.uniform(0.80, 1.20)
        price_hint = int(round(max(1.0, hint)))

        # hidden true price = base * unseen per-round multiplier in ~[0.7, 1.4].
        mult = rng.uniform(0.70, 1.40)
        true_price = int(round(max(1.0, base * mult)))

        rounds.append((value, price_hint, true_price))
        total_price += true_price

    # budget: a fraction of the total true price, so only a minority is affordable.
    frac = rng.uniform(0.20, 0.45)
    B = max(1, int(round(frac * total_price)))

    out = [f"{T} {B}"]
    for (value, hint, tp) in rounds:
        out.append(f"{value} {hint} {tp}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
