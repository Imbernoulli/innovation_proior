# Skew Ahead: Quoting a Market Book Against Adverse Scripted Order Flow

You run a market maker on one asset over `T` discrete steps. At each step `t` you post a
two-sided quote around the public mid price `m_t`:

- a **bid** a distance `hb_t ≥ 0` below mid — you *buy* there and grow **long**;
- an **ask** a distance `ha_t ≥ 0` above mid — you *sell* there and grow **short**;

each backed by a maximum quoted **size** (`zb_t` on the bid, `za_t` on the ask).

## Scripted order flow (it is adversarial)

Every step a deterministic, seeded flow script releases `S_t` units of marketable **sell**
interest (which can hit your bid, so **you buy**) and `B_t` units of marketable **buy**
interest (which can lift your ask, so **you sell**). Only a fraction actually trades against
you, and that fraction **rises as your quote tightens**:

- units filled on the bid: `S_t · max(0, 1 − hb_t / W)`, capped by your size `zb_t`;
- units filled on the ask: `B_t · max(0, 1 − ha_t / W)`, capped by your size `za_t`.

The script leans its interest **into** each coming move: a burst of buy interest ahead of an
up-move (lifting your ask, forcing you short right before the price rises), a burst of sell
interest ahead of a down-move. It tries to hand you exactly the inventory the move will punish.

## Inventory carries risk

Fills change a signed inventory `q` that **carries over** across steps and is clamped to
`[−Qmax, +Qmax]` (a fill that would breach the cap is truncated). Each step you pay a convex
holding charge `lam · q²` on the inventory you are left carrying, and a terminal charge
`mu · q_T²`. Your book is marked to the final mid `m_T`.

Buys move cash by `−qb·(m_t − hb_t)`; sells by `+qs·(m_t + ha_t)`.

## Objective (maximize)

```
PnL = cash_T + q_T · m_T  −  lam · Σ_t q_t²  −  mu · q_T²
```

Quoting nothing (never filling) yields `PnL = 0`.

## Input (one JSON object on stdin — the public instance)

```
{"name": str, "T": int, "W": float, "Qmax": float, "lam": float, "mu": float,
 "m": [m_0 … m_T],          # T+1 mid prices — the move is visible in advance
 "S": [S_0 … S_{T-1}],      # sell interest per step (hits your bid)
 "B": [B_0 … B_{T-1}]}      # buy interest per step (lifts your ask)
```

The whole mid path and both flow series are public, so every move is **anticipated**. The
holding coefficients, cap, and fill depth `W` live in the input — read and exploit them.

## Output (one JSON object on stdout)

```
{"hb": [ … T … ], "ha": [ … T … ],   # bid / ask half-spreads, each ≥ 0
 "zb": [ … T … ], "za": [ … T … ]}   # bid / ask quoted sizes,   each ≥ 0
```

Each list must have exactly `T` finite, non-negative numbers (no `NaN`/`inf`/negative). Any
violation, crash, timeout, or non-JSON scores 0 on that instance.

## Scoring (deterministic; no wall-time)

For each of 10 fixed seeded instances your `PnL` is normalized against a **loose, unreachable
upper bound** built from that instance's own parameters (full directional capture at the cap on
every move plus the textbook maximum spread capture on all flow):

```
r = clamp( 0.1 + 0.9 · PnL / hi ,  0, 1 )
```

Do-nothing scores exactly `0.1`; the slack in `hi` keeps even a well-positioned book below
`1.0`. Your score is the **mean of `r`** over the 10 instances, which span adverse single-move
traps, calm spread-capture books, and held-out twin-move (up-then-down) regimes.

## What to notice

Posting the spread that maximizes myopic per-step capture (tight, symmetric, full size) lets
the adverse burst load you the wrong way and you bleed the move for a sliver of spread. Because
the price path is public you can instead choose a **target inventory that leans the way the move
pays**, invert the fill formulas to reach it on one side while withdrawing the other, build the
position **early** (the pre-move burst chokes the interest you need), and unwind afterward to
shed the convex holding charge. Spread and terminal inventory risk trade off against each other.
