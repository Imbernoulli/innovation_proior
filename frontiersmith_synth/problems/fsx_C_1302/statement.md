# Bidding Against Bots That Watch You Back

## Setting

You are a demand-side bidder facing a fixed **sequence of `T` back-to-back
auctions** (rounds `0..T-1`), each for one item. You have a single **hard
budget** `budget` that never replenishes across the whole sequence.

For round `i` you are given a **public signal** `base[i] >= 0` -- your best
point estimate of how valuable item `i` is. The item's *true* value is
`base[i] * mult[i]`, where `mult[i]` is a hidden per-round multiplier you
never see: the signal is informative but noisy, so you cannot know exactly
what any round is really worth before it resolves.

Every round also has exactly one **reactive competitor** with a public
baseline bid `comp_base[i]`. The competitor does not bid `comp_base[i]`
outright -- it watches how aggressively *you* have been bidding and
ratchets its own bid up in response:

```
recent_avg = average of (your effective bid / base[j]) over the last
             mem_k rounds j < i that have already happened (0 if none yet)
competitor_bid[i] = comp_base[i] * (1 + adapt_rate * recent_avg)
```

So a round of heavy bidding trains the competitor to bid higher for the
rounds that follow, for a window of `mem_k` rounds.

## Round mechanics

You submit one bid `x[i] >= 0` per round, ahead of time, for the whole
sequence (see Output below). Rounds are then replayed in order `0..T-1`:

1. Your bid is clipped to whatever budget remains: `eff = min(x[i],
   remaining_budget)`. You can never spend more than you have left.
2. If `eff >= competitor_bid[i]` and `eff > 0`, you **win** round `i` and
   pay the competitor's bid (a second-price rule: you pay what the
   next-highest bidder offered, *not* your own bid). `remaining_budget`
   drops by that price. Your surplus for the round is `true_value[i] -
   price`. Otherwise you win nothing and pay nothing (surplus 0).
3. Whether or not you win, `eff / base[i]` is recorded as this round's
   *aggression* and feeds `recent_avg` for the next `mem_k` rounds.

Because the price you pay is the competitor's bid (not yours), bidding
higher than necessary to win never costs you more THIS round -- but it does
feed a higher aggression signal that raises the competitor's bids in
*future* rounds. Your **total surplus** is the sum of per-round surpluses
over all `T` rounds; you want to **maximize** it.

## Input (stdin, one JSON object)

```
{"name": str, "T": int, "budget": float,
 "base": [float]*T, "comp_base": [float]*T,
 "adapt_rate": float, "mem_k": int}
```

## Output (stdout, one JSON object)

```
{"bids": [float]*T}
```

Exactly `T` numbers, each finite and `>= 0`. Any wrong length, non-numeric
entry, negative value, `nan`/`inf`, a crash, or a timeout scores `0.0` for
that instance.

## Scoring

The checker replays the exact mechanics above using your bid vector and the
HIDDEN true values to compute your total surplus `F`. Two internal
references (never revealed to you) anchor the score for each instance: a
do-nothing baseline (`F=0`) anchored to `0.1`, and an oracle that sees the
TRUE values and prices every round at its worst-case fully-escalated
competitor cost, then greedily funds the rounds with the highest surplus
per dollar of that cost until the budget runs out (anchored to `1.0`).
Formally, with `F_ideal` the oracle's surplus:

```
score = clamp(0.1 + 0.9 * F / F_ideal, 0, 1)
```

averaged over all instances. The oracle has strictly more information than
you (exact values, not a noisy signal), so matching it exactly is generally
not possible.

## Constraints

- `30 <= T <= 70`, `10` instances, values/costs are non-negative floats
  bounded well under `1e4`.
- Deterministic: everything is seeded; no randomness, wall-clock, or I/O
  order affects the score.
- Time limit: 5s per instance. Memory: 512MB.

## Notes

Some instances put the valuable rounds early with a generous budget (timing
barely matters there); others put them late with a tight budget and a fast
adaptation rate, so spending on the early, unremarkable rounds both drains
your budget and trains the competitor before the rounds that actually
matter arrive. Read `base`, `comp_base`, `adapt_rate` and `budget` and plan
the whole sequence accordingly -- there is no single fixed round to "start
bidding hard" that works across every instance.
