# Firebreaks Against an Unpredictable Blaze

## Story

You manage fire risk on an `R x C` grid of forest cells. Each cell is either
**flammable** (forest) or already bare ground / rock (non-flammable). Before
the season starts you get to convert a limited **budget** of currently
flammable cells into firebreaks — permanently removing them from the
flammable footprint — and then the season plays out with no further
intervention from you.

You are handed two historical hints: a **hot zone**, the block of cells that
has ignited most often in past seasons, and this season's **prevailing
wind** direction. Neither hint is destiny. A fixed fraction of ignitions
really do start in the hot zone, but the rest start **anywhere** in the
flammable footprint. And even an ignition in the hot zone only travels with
the prevailing wind about half the time — the rest of the time the wind
blows some other way.

Once ignited, fire spreads over the surviving flammable cells as a
**directed percolation process**: from a burning cell, each flammable,
non-firebreak neighbor independently catches fire at a probability set by
that day's wind — higher downwind, lower upwind, baseline on the two
perpendicular directions. Above the percolation threshold, a large flammable
footprint forms **one giant connected cluster** — a fire starting anywhere
inside it can, in principle, reach almost the entire cluster, regardless of
which corridor that day's wind happens to favor. The size of the piece a
fire's starting point ends up trapped inside — not which single path out of
the hot zone you blocked — is what caps the damage.

## Input (stdin, one JSON object)

```json
{ "R": 14, "C": 18, "flammable": [[1,0,1,...], ...],
  "budget": 14, "hint_zone": {"r0":2,"r1":4,"c0":2,"c1":4},
  "wind_bias": [0, 1], "p_base": 0.75, "wind_extra": 0.20,
  "p_wind_dominant": 0.5, "w_hint": 0.40, "n_draws": 30 }
```
- `flammable[r][c]` is 1 (flammable) or 0 (already bare ground), row-major.
- `hint_zone` cells are always flammable.
- `wind_bias` is `(dr, dc)`, one of `(-1,0)/(1,0)/(0,-1)/(0,1)`.
- Directed spread probabilities: downwind edges open at `p_base +
  wind_extra`, upwind edges at `p_base - wind_extra`, the two perpendicular
  directions at `p_base`.
- `p_wind_dominant`: probability a given ignition's wind equals `wind_bias`
  (else uniform over the other three directions).
- `w_hint`: probability a given ignition lands inside `hint_zone` (else
  uniform over every originally-flammable cell on the board).
- `n_draws`: how many hidden ignition/wind scenarios your placement is
  scored against (the realizations themselves are never shown to you).

## Output (stdout, one JSON object)

```json
{ "cells": [[2,5], [3,5], [4,5], ...] }
```
A list of at most `budget` **distinct** cells, each in bounds and flammable
(`flammable[r][c]==1`) in the input grid — the cells you convert to
firebreak. Any duplicate, out-of-bounds, non-flammable, over-budget, or
malformed answer, a crash, a timeout, or non-JSON output scores this
instance `0.0`.

## Scoring (deterministic)

The evaluator removes your cells from the flammable graph, then replays a
fixed family of hidden ignition/wind scenarios drawn from the mixture above,
flooding each one out over the surviving flammable cells with the directed
percolation rule. Your objective for the instance is:

```
obj = mean over scenarios of (cells burned / total originally-flammable cells)
```

which you **minimize**. Let `b` be the same quantity for an **empty**
placement (do nothing) — the evaluator computes this directly. Your
instance score is

```
r = min(1.0, 0.1 * b / max(obj, 1e-12))
```

so doing nothing reproduces the baseline exactly (`r = 0.1`); every
proportional reduction in burned fraction below the do-nothing baseline
raises `r` toward 1.0, and doing no better (or worse) than baseline keeps
`r <= 0.1`. Your final score is the mean of `r` over a fixed family of 10
boards of varying shape, density, hint-zone placement, and wind.

## Notes

- Everything is seeded and deterministic — no wall-clock timing, no hardware
  dependence, and re-running gives identical scores.
- Your program is called once, open-loop, before any ignition is drawn; it
  runs sandboxed and only ever sees the JSON above. The hidden draws and the
  do-nothing baseline are computed only in the evaluator process.
- No free lunch: a firebreak wall only helps scenarios whose fire actually
  tries to cross it. A budget spent walling off the single corridor the
  hints point at leaves the rest of a large, connected flammable footprint
  exactly as reachable as before.
