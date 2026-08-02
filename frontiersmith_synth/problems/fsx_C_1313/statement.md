# Kiln Firing Policy: Gate the Ramp by the Thickest Piece

## Story

A kiln load contains several ceramic pieces of *different wall thickness*, fired together
on **one shared temperature-vs-time schedule** (the kiln only has one burner; every piece
sees the same surface/kiln temperature at every instant). You must submit that schedule.

Each piece's **core** temperature lags the kiln's surface temperature. The lag time
constant grows with the *square* of the piece's wall thickness (standard heat-diffusion
scaling) — a thick piece's core falls much further behind than a thin piece's.

The clay body's silica undergoes abrupt **phase inversions** at specific temperature
*bands* (quartz around 555-600 °C, cristobalite around 205-245 °C; the exact numbers are in
the input). **Inside** a band, a surface-core temperature *gradient* converts into
cumulative thermal-shock stress (the effect grows with the *square* of the gradient).
**Outside** any band, the material tolerates any gradient for free — speed there costs
nothing. If a piece's cumulative in-band stress exceeds its own crack threshold, its value
degrades linearly, reaching zero once stress reaches 2x threshold. Firing also costs fuel
proportional to total minutes elapsed, so idling forever is not free either.

Heating as fast as the burner allows minimizes fuel time — but it drives whichever piece is
*thickest* so far behind the surface while crossing a band that it cracks. The schedule only
needs to slow down **inside the bands**, and only by however much the **single thickest
piece** in the load demands; every other piece is automatically safe.

## Isolation

Your program runs as an **isolated subprocess**: it reads one JSON object (the full public
instance — nothing is hidden here, this is a fully-specified optimization problem) from
**stdin** and writes one JSON value (your schedule) to **stdout**.

## Public instance (stdin)

```json
{
  "start_temp": 20.0, "target_temp": float, "max_rate": float,
  "max_total_minutes": float, "max_segments": int, "sim_dt_minutes": float,
  "fuel_cost_per_minute": float,
  "bands": [ {"lo": float, "hi": float, "multiplier": float, "name": str}, ... ],
  "pieces": [ {"thickness_mm": float, "value": float, "fragility": float}, ... ],
  "diffusion_k": float,
  "stress_threshold_k": float
}
```
`tau_i` (minutes, core-lag time constant of piece `i`) `= diffusion_k * thickness_mm_i^2`.
Piece `i`'s crack threshold `= stress_threshold_k * fragility_i`.

## Answer (stdout)

A JSON list of ramp segments (a bare `{"schedule": [...]}` wrapper is also accepted), each
`{"to_temp": float, "minutes": float}`, meaning: move the kiln surface linearly from
wherever it currently is to `to_temp` over the next `minutes` minutes (`minutes > 0`).
`to_temp` must be **non-decreasing** across segments (firing only heats up), and the implied
rate `(to_temp - prev)/minutes` must never exceed `max_rate` (a rate of 0 over a positive
duration means the kiln is held level — a hold). The schedule must reach `target_temp`;
simulation stops the instant it does, so extra segments afterward don't matter for timing
but must still be well-formed. Total simulated minutes must not exceed `max_total_minutes`.
Any violation (wrong shape, non-finite numbers, decreasing `to_temp`, rate over cap, never
reaching target, exceeding the time cap, a crash, or a timeout) scores that instance **0**.

## Scoring

Each piece's core temperature is the exact solution of `dcore/dt = (T_surf(t) - core)/tau_i`
(no discretization error). A piece's in-band stress accumulates
`(T_surf(t) - core(t))^2` only while that moment's temperature lies inside a band, scaled by
that band's `multiplier`. `raw_obj = sum(surviving piece values) - fuel_cost_per_minute *
total_minutes`.

The evaluator normalizes `raw_obj` against two schedules it builds itself: a **baseline**
(constant max-rate the whole way) and a **ceiling** (fast outside bands, gated only inside
them by the thickest piece, with headroom left over the ceiling too):

```
r = clamp( 0.1 + 0.9 * (raw_obj - obj_base) / (obj_ceiling - obj_base), 0, 1 )
```

so matching the naive fast schedule maps to ≈0.1. `Ratio` is the mean of the 10 per-instance
`r` values.

## Objective

**Maximize `Ratio`.** A flat compromise rate wastes fuel everywhere for safety it only
needed in two narrow bands; the fastest-possible rate saves fuel but cracks the load on
mixed-thickness kilns. The schedule that goes fast everywhere *except* where — and only
because — the thickest piece demands it wins on both fuel and intact ware.
