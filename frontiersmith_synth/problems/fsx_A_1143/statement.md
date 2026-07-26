# Three Hundred Lots: The Ledbury Room

You are an antique buyer seated in the Ledbury Room for a single session of
**300 sequential lots**, sold one at a time by **sealed-bid second-price
auction**. Three other bidders share the room. Each of them is a **published,
deterministic, budget-paced rule-follower** — their exact bidding formulas
are printed below, with per-instance constants supplied in the input. Nothing
about them is hidden or random: given the input, you can replay the whole
session in your head before the gavel falls once.

On every lot, all four bidders submit sealed bids simultaneously. The
**highest bid wins** and pays the value of the **second-highest bid** (a
losing bid still sets the price if it is the second-highest). Ties are
broken deterministically: **you (the buyer) lose every tie** against any
opponent; among opponents, ties go to **pacer, then sniper, then capper** (in
that fixed order). A bidder's remaining budget decreases **only when that
bidder wins** a lot, by exactly the price paid — a losing bid never costs its
owner anything, but it can still cost the *winner* extra, since it may become
the new second-highest price.

## Opponent formulas (published)

Let `E_i` be lot `i`'s catalogue estimate, `cat_max` the maximum estimate
over the whole catalogue (all 300 estimates are visible up front), and
`rem_lots = 300 - i` (lots left including this one). Each opponent bids from
its **own current remaining budget** `rem`:

- **pacer** (`budget`, `k`): `share = rem / rem_lots`; bid `= k * min(share, E_i)`.
- **sniper** (`budget`, `thresh`, `spike`, `low`): if `E_i >= thresh * cat_max`
  (a marquee lot) bid `= spike * E_i`; else bid `= low * E_i`.
- **capper** (`budget`, `cap_mult`, `cap_frac`): bid `= min(cap_mult * E_i,
  cap_frac * budget0)`, where `budget0` is that opponent's *starting* budget
  (its `budget` field, unchanged in the input).

Every opponent's raw bid is finally clamped to `[0, rem]`.

## Candidate program contract

Standalone program, stdin → stdout, isolated subprocess:
```python
import sys, json
inst = json.load(sys.stdin)
# ...compute...
print(json.dumps({"bids": bids}))
```

**Public instance (stdin)**
```json
{"name": "session1001", "n_lots": 300, "buyer_budget": 21000.0,
 "lots": [{"estimate": 137.4, "value": 151.9}, ...],
 "opponents": [
   {"type": "pacer",  "budget": 27000.0, "k": 0.85},
   {"type": "sniper", "budget": 21000.0, "thresh": 0.45, "spike": 3.6, "low": 0.12},
   {"type": "capper", "budget": 26000.0, "cap_mult": 1.2, "cap_frac": 0.05}
 ]}
```

**Answer (stdout)**
```json
{"bids": [b_0, b_1, ..., b_299]}
```
`bids` must have exactly `n_lots` finite, non-negative numbers — your sealed
bid on every lot, all submitted up front (you know everything needed to plan
the whole session in advance).

You are **not** required to keep any bid within your own remaining budget —
you may bid above what you could pay, betting the formulas above guarantee
you will not win. If a bid **does** win a lot and you cannot cover the price
from your remaining budget, the answer is rejected: the **whole session
scores 0**. Any malformed answer (wrong length, non-numeric, negative,
NaN/Inf), a crash, or a timeout also scores 0.

## Objective and scoring (deterministic)

Your utility is `sum over lots you win of (your value - price paid)`. We
normalize against a fixed, loose anchor computed by the evaluator itself:

```
U_ref = sum(value_i for all lots)          # unreachable: winning everything free
r = clamp(0.1 + 0.9 * utility / U_ref, 0, 1)
```

Winning nothing scores exactly `0.1`; `U_ref` is far out of reach (three
funded, competing bidders and your own finite budget), so real sessions stay
well under `1.0`. The reported **Ratio** is the mean `r` over 10 seeded
sessions; **Vector** lists the per-session scores.

## What to think about

A bid on a lot you don't want isn't wasted: since only the *winner* pays, and
you always lose ties, bidding **exactly** an opponent's computed bid is a
completely safe "bluff ceiling" — you never win, but you force whoever does
win to pay their own top bid in full instead of the cheaper price they would
otherwise have faced, draining a specific opponent's budget for every lot
that follows. Since every rule above is fully known, this is a planning
problem, not a guessing game.
