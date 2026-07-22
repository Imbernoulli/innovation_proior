# Ten Rounds to Close: Multi-Supplier Procurement Under Timed Concessions

You run procurement for a plant that needs a total quantity **Q** of one commodity.
You have **M** seeded suppliers and one always-available **outside option** (a
backstop spot-market source). You have at most **T** negotiation rounds to secure
all of Q; write a program that plans the whole T-round campaign.

## Suppliers and concession curves

Supplier `i` posts an opening ask `p0_i` and will never sell below its floor ask
`pfloor_i`. Its current ask is governed by a **concession level** `c_i`, starting at
0 and capped at 1:

    ask_i(c_i) = pfloor_i + (p0_i - pfloor_i) * (1 - c_i)

Each round t (1-indexed) you take exactly one action:

- **negotiate** supplier `i` (only legal while `t <= deadline_i`): this advances
  its concession, `c_i += step`, where `step = base_step_i`, multiplied by
  `soften_mult_i` (a **softening** bonus) whenever the round is within `window_i`
  rounds of that supplier's *own* deadline (`deadline_i - t < window_i`). In the
  same round you may also buy `qty` units from `i` at the resulting ask, bounded
  by `i`'s remaining capacity `cap_i` and your remaining need.
- **outside**: buy `qty` units immediately at `outside_price(t) = outside0 *
  outside_growth^(t-1)`, no negotiation needed. Since `outside_growth > 1`, this
  price only ever gets worse the longer you wait -- the outside option **decays**.
- **wait**: do nothing this round.

**Hardening.** Any supplier you do *not* negotiate with this round, but that you
had previously engaged (`c_i > 0`), *regresses* because you walked away:
`c_i -= harden_step_i` (floored at 0). This applies to every non-negotiated
supplier every round, including rounds where you use the outside option or wait.

Any need still unmet after T rounds is force-filled at a penalty price
`outside_price(T+1) * penalty_mult` (always worse than any live outside price).

## Input (stdin, one JSON object)

```
{"T": int, "Q": float, "M": int,
 "suppliers": [ {"p0":float,"pfloor":float,"base_step":float,"harden_step":float,
                 "soften_mult":float,"window":int,"deadline":int,"cap":float}, ... ],
 "outside0": float, "outside_growth": float, "penalty_mult": float}
```
All M suppliers' full parameters are given upfront -- this is a full-information
planning problem, not a discovery/bandit problem.

## Output (stdout, one JSON object)

```
{"actions": [a_1, a_2, ..., a_k]}     # k <= T rounds; missing trailing rounds = wait
```
Each `a_t` is one of:
```
{"type":"negotiate", "supplier": int, "qty": float}   # qty optional, default 0
{"type":"outside", "qty": float}
{"type":"wait"}
```
Any malformed shape, out-of-range supplier index, or non-finite/negative `qty`
invalidates the WHOLE answer for that instance (score 0 there).

## Scoring

The evaluator replays your action list against the round loop above and computes
your total procurement cost. It compares this against two references it computes
itself:

    cost_base = Q * outside0                    # buy everything from outside at round 1
    cost_ub   = Q * min_i(pfloor_i)              # optimistic, generally unreachable:
                                                  # every unit at the single best floor,
                                                  # ignoring capacity/deadlines/rounds-needed
    score = clamp( 0.1 + 0.9 * (cost_base - your_cost) / (cost_base - cost_ub), 0, 1 )

Doing nothing (buy it all from outside immediately) scores ~0.1. The unreachable
all-floor bound scores 1.0. Your final score is the mean over 10 instances,
several of which are held out / larger to test generalization.

## Strategy notes

Chasing whichever supplier quotes the best price *this round* is a trap: a
supplier with an unappealing opening ask can still have the best FLOOR and the
fastest concession -- but a one-round-lookahead policy never "pays" the costlier
early rounds needed to discover that, and it hardens every supplier it neglects
along the way. Likewise, treating the outside option as a last resort wastes its
one advantage: it is cheapest right now and only decays from here, so any
shortfall you already know suppliers cannot cover should be locked in early, not
left to the end.
