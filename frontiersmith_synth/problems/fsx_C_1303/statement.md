# Rebalancing a Line While It Is Running

## Story

A serial assembly line has `K` stations in a row, station `0, 1, ..., K-1`,
connected by `K-1` finite WIP buffers (buffer `j` sits between station `j`
and station `j+1`). Raw material is infinite before station 0 and shipped
units leave freely after station `K-1`. The line runs for `T` product-mix
**epochs**; epoch `t` lasts `L[t]` ticks. In epoch `t`, station `i`'s *base*
cycle time (time per unit with no boost) is `base_cycle[t][i]` — because the
product mix changes every epoch, **which station is slowest migrates** across
the horizon: sometimes drifting smoothly, sometimes oscillating between two
or three stations.

A shared pool of `P` **boost units** (extra labor/tooling) can be
redistributed across stations each epoch. Assigning `u` boost units to
station `i` divides its cycle time by `(1 + k_eff[i]*u)` — stations differ in
how much boost actually helps them. Re-pointing boost units at a station is a
**changeover**: any station whose boost allocation changes from the previous
epoch is *offline* (produces nothing, neither pulls input nor pushes output)
for the first `d0[i] + d1[i]*|delta|` ticks of the new epoch (capped so at
least 1 tick of real production remains), and a booking/logistics cost
`m0[i] + m1[i]*|delta|` is charged. WIP buffer sizes between stations are a
**one-time** decision from a fixed total `buffer_budget` (physical shelving,
not re-planned every epoch) — bigger buffers let the line absorb a transient
imbalance (a changeover stall, or a station suddenly becoming the bottleneck)
without a full stop.

You are given the **entire** epoch schedule up front — nothing about the
instance is hidden from you.

## Input (stdin, one JSON object)

```json
{
  "K": int, "T": int, "P": int, "buffer_budget": int,
  "L": [int, ...T],
  "base_cycle": [[float, ...K], ...T],
  "k_eff": [float, ...K],
  "changeover_downtime_fixed": [float, ...K],
  "changeover_downtime_per_unit": [float, ...K],
  "changeover_money_fixed": [float, ...K],
  "changeover_money_per_unit": [float, ...K],
  "initial_alloc": [int, ...K],
  "money_weight": float, "seed": int
}
```

`initial_alloc` is the boost allocation the line starts the horizon with
(before epoch 0) — reallocating away from it at epoch 0 also pays a
changeover.

## Output (stdout, one JSON object)

```json
{"alloc": [[int, ...K], ...T],   // boost units per station, per epoch
 "buffers": [int, ...K-1]}       // one-time buffer capacity per gap
```

Every `alloc[t]` row must be `K` nonnegative integers summing to `<= P`.
`buffers` must be `K-1` integers, each `>= 1`, summing to `<= buffer_budget`.
Any violation (wrong shape, non-finite, over budget, non-integer) scores that
instance **0**, as does a crash, timeout, or missing output.

## Scoring

The evaluator re-simulates your policy tick-by-tick (a deterministic flow
model: each tick, stations are processed downstream-to-upstream, each
station moving `min(its rate, available upstream input, available downstream
buffer space)` units, `0` while in changeover) to get `shipped` (units
leaving station `K-1` over the whole horizon) and `money` (total changeover
booking cost). Your instance objective is `shipped - money_weight * money`.

This is affine-normalized against two reference policies the evaluator
computes the same way: `obj_base` (keep `initial_alloc` forever, uniform
buffer split — zero changeover) and `obj_ref` (an unreachable upper bound:
re-optimize the allocation every epoch for **free**, with infinite buffers).

```
r = clamp(0.1 + 0.9 * (your_obj - obj_base) / (obj_ref - obj_base), 0, 1)
```

`Ratio` is the mean of the 10 per-instance `r`.

## Objective

**Maximize `Ratio`.** Chasing whichever station looks slowest *this epoch*
is easy to write but pays repeated changeovers, can target a station that
barely benefits from boost, and leaves buffers unable to absorb the next
migration. A policy that reads the *whole* schedule, decides where the
changeover is actually worth paying for, and sizes buffers for the
migrations it foresees does substantially better.
