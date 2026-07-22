# Tidewater Berth: Admission Under Value Drift

## Story

A single deep-water berth offers a fixed pool of `B` mooring credits for one tide
window. Over the window, `T` vessels arrive **one at a time**. Each vessel has a
**class** `c ∈ {0,1,2}` (coastal, regional, deep-sea), a **size** `w` (mooring
credits it locks up for the rest of the window — credits are **never released**),
and a **fee** `v` it pays if admitted. The harbor master must decide **ADMIT** or
**DECLINE** the instant a vessel arrives — irrevocably — and total admitted size
may never exceed `B`.

The twist: each class's fee distribution **drifts at a hidden mid-window
breakpoint** (a tariff regime change). On several tides the lucrative **deep-sea**
class only appears *after* the drift, so a bar fitted to the calm early water
spends every credit on cheap coastal traffic and turns the whales away.

You never see future fees — a vessel's fee is revealed only when it arrives, and
you are told **only the early-regime** fee means. Instead of guessing fees, you
**ship a causal admission policy** and the harbor runs the tide against it.

## What you submit

Read one PUBLIC instance (JSON) from stdin and write one JSON object to stdout:

```
{"bars": bars}      # bars[c][rb][sb] : float, shape [K][R_buckets][S_buckets]
```

`bars[c][rb][sb]` is the **minimum fee** you will admit for a class-`c` vessel when
the berth is in remaining-bucket `rb` and drift-signal bucket `sb`. The evaluator
runs the tide **causally**, revealing each fee only at its own arrival, and admits
vessel `t` iff

```
fee_t >= bars[class_t][rb_t][sb_t]   AND   size_t <= remaining_credits
```

where, at arrival `t`:

- `rb_t = min(R_buckets-1, floor(R_buckets * remaining / B))` — how full the berth is
  (`rb=0` almost empty of credits, `rb=R_buckets-1` almost full).
- `sb_t` = drift probe: let `m` be the running mean of the **last `window`** revealed
  fees (up to and including `t`). Count how many `sig_edges` values `m / prior_g`
  meets or exceeds; that count (capped at `S_buckets-1`) is `sb_t`. A rising `sb`
  means the tariff regime has drifted **up**.

All of `B, T, K, classes, sizes, prior_mu, prior_g, R_buckets, S_buckets, window,
sig_edges` are in the PUBLIC instance. You see the full **class/size schedule** and
the **early-regime** means `prior_mu` (and their global mean `prior_g`) — but never
the fees, the breakpoint, or the drifted late-regime means.

## PUBLIC instance schema

```
{"B": int, "T": int, "K": 3,
 "classes": [c_0..c_{T-1}], "sizes": [w_0..w_{T-1}],
 "prior_mu": [mu0, mu1, mu2],           # early-regime class fee means (published)
 "prior_g": int,                         # published early global mean fee
 "R_buckets": int, "S_buckets": int, "window": int, "sig_edges": [floats]}
```

## Answer schema

```
{"bars": [[[float]*S_buckets]*R_buckets]*K}
```

A `bars` that is not exactly this shape, or contains a non-finite number, scores
`0.0` on every tide.

## Objective (maximize)

Let `v_cand` be the total fee your policy admits. The evaluator computes two
references on the **hidden** fees: `v_open` (open-door: admit-if-it-fits in arrival
order) and `v_opt` (offline **fractional** optimum — fractional knapsack on hidden
`fee/size` with full hindsight, an optimistic upper bound). Each tide scores

```
r = clamp( 0.1 + 0.9 * (v_cand - v_open) / max(1e-9, v_opt - v_open), 0, 1 )
```

Open-door scores ≈ 0.1; matching the (generally unreachable) offline optimum scores
1.0; doing worse than open-door scores below 0.1. The final score is the mean `r`
over all tides. Because the late tariff is hidden and `v_opt` is a loose hindsight
bound, even a strong reserving policy stays well below 1.0 — there is headroom.

## Why the obvious bar is a trap

Fitting a flat per-class bar to `prior_mu` ignores both **how full the berth is**
and the **drift signal**: it admits cheap early traffic, exhausts credits, and has
none left when the scarce deep-sea class finally arrives after the drift. Turning
the fee gate into a **shadow price** — reserving credits for the demand you can see
coming and releasing the bar once the drift is probed — captures exactly the
vessels the flat bar misses.

Determinism: every tide is seeded; identical submissions score identically. Time
limit 2–5 s per instance, memory ≤ 512 MB.
