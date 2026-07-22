# Plumbing the Village Above the Permafrost

## Story

Your village sits on a grid of `rows x cols` cells, each with a ground-cover
type: **bare soil**, **plowed pavement**, **snow-holding open field**, or
**insulating peat/bog**. `K` buildings need a single connected water-pipe
network. You must choose exactly which cells the pipe occupies (it must form
one 4-connected piece touching every building) and, for every cell it
occupies, the burial depth. Digging is expensive and **quadratic**: a cell
buried at depth `d` costs `d^2` to excavate. Bury too shallow anywhere,
though, and that cell's water freezes solid in some recorded winter --
catastrophic, not a partial loss.

## Frost physics (exactly as the evaluator computes it -- reproduce it)

For cover type `t` and a historical winter `w` with freezing-degree-days
`fdd_w` and seasonal snowfall `snow_w`:

```
kappa_eff(bare, w)     = kappa.bare
kappa_eff(pavement, w) = kappa.pavement
kappa_eff(snow, w)     = kappa.snow_base / (1 + kappa.snow_sensitivity * snow_w)
kappa_eff(peat, w)     = kappa.peat
frost_depth(t, w)      = depth_scale * sqrt(2 * kappa_eff(t, w) * fdd_w)
required_depth(t)      = max over ALL given winters w of frost_depth(t, w)
```

A pipe cell of type `t` must be buried at depth `>= required_depth(t)` (a
tiny floating-point tolerance is allowed) or the instance scores 0. Because
`snow`'s insulation depends on that winter's *snowfall*, its worst winter is
often a mild-but-dry one, **not** the coldest winter on record -- unlike the
other three types, whose worst case is always simply the highest-`fdd`
winter. `pavement` is the most exposed cover (largest kappa); `peat` is
consistently the best insulator; `bare` and `snow` fall in between and vary
by winter.

## Input (one JSON object on stdin)

```json
{"name": "village03", "rows": 14, "cols": 14,
 "grid": [[0,1,3,...],[...],...],
 "buildings": [[1,1],[1,12],[12,1],[12,12]],
 "winters": [{"fdd": 1423.7, "snow": 0.91}, ... 10 entries ...],
 "kappa": {"bare":1.0,"pavement":1.55,"snow_base":1.0,"snow_sensitivity":0.9,"peat":0.38},
 "depth_scale": 0.03,
 "type_names": {"0":"bare","1":"pavement","2":"snow_holding","3":"peat"}}
```
`grid[r][c]` is one of `0,1,2,3` (row-major, `rows` rows x `cols` cols).
`buildings` lists `K` `[r,c]` cells (3-5 depending on instance) that all
must end up in your pipe network.

## Output (one JSON object on stdout)

```json
{"route": [{"r": 1, "c": 1, "depth": 1.9}, ... one entry per occupied cell ...]}
```
Every `(r,c)` must be a distinct in-bounds cell; `depth` a finite number in
`[0, 50]`. The set of `(r,c)` cells must form a single 4-connected component
and must contain every building. Any malformed output, duplicate cell,
missing building, disconnection, or freeze (declared depth below that
cell's `required_depth`) scores that instance `0.0`.

## Scoring (deterministic, no wall-time)

Excavation cost = `sum(depth^2)` over every occupied cell. The evaluator
computes two references itself, never sent to you: `cost_naive` (a
straight-line chain route buried everywhere at one deliberately
over-conservative flat depth -- the "safe and lazy" recipe) and `cost_ref`
(a strong internal lower reference built from a warped-metric routing
search, scaled down to leave headroom). Your instance score is

```
r = clamp(0.1 + 0.9*(cost_naive - cost) / max(1, cost_naive - cost_ref), 0, 1)
```

Matching the naive recipe scores about `0.1`; a route that is smart about
depth but still geometry-blind scores clearly higher; a route that
genuinely reroutes through protective ground to shrink the *quadratic* bill
scores higher still. Your final score is the mean of this `r` over 10
instances (grid size, building count, and terrain layout vary; some are
noticeably more forgiving of a naive route than others).

## Notes

- The straight geometric shortest route between two buildings is frequently
  **not** the cheapest to bury: `pavement`'s required depth is roughly
  double `peat`'s, and cost is squared in depth, so a considerably longer
  detour through peat or (in the right winter mix) snow-holding ground can
  cost a fraction of the direct route.
- Your program runs in an isolated subprocess and sees only the JSON above;
  scoring never depends on wall-clock time.
