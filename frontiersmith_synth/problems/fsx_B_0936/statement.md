# Stock Stations Before Bikes Run Out (Format B, isolated)

A bike-share network has `n` stations, each with a fixed dock capacity. You
must choose how many bikes to place at every station **before** the day
starts, spending from one shared total-bike **budget** — you cannot exceed
any station's dock capacity, and you cannot spend more bikes than the
budget allows.

A **frozen, seeded trip schedule** is then replayed against your choice.
Every trip is `(t, o, d, dt)`: a rider requests a bike at station `o` at
time `t`; if one is available, it is taken immediately and the trip arrives
at station `d` at time `t + dt`. Two things can go wrong, and both are
**saturating** — there is no partial credit, no queue, no backorder:

- **Stockout.** If station `o` has zero bikes at time `t`, the trip is
  simply **lost** — the rider walks away. (A station that later regains a
  bike does not retroactively help a trip that already failed.)
- **Overflow.** If, when the trip's bike arrives at `d`, every dock at `d`
  is already full, the bike **cannot dock** — it is **stranded** and
  removed from the system permanently (that bike never becomes available
  again, anywhere, for the rest of the day).

Ties are broken deterministically: at any timestamp, pending arrivals are
applied *before* that timestamp's departures (a bike arriving exactly when
another trip wants to leave is available for it); departures sharing a
timestamp are attempted in `trips`-array order (earlier entries first).
After the schedule ends, bikes still in transit are delivered (same
overflow rule).

The stations are **coupled**: a bike leaving one station is the bike that
later succeeds or fails at another. A station's right buffer size depends
on the full timing of its trips — not simply "how much net demand does
this station have" — since a station can send and receive nearly equal
traffic in total while still needing a large buffer if its outflow
happens well before its matching inflow arrives (or vice versa, needing
to stay emptier to avoid overflow).

## Public instance (stdin JSON)
```json
{
  "n": 10,
  "capacity": [c_1, ..., c_n],        // dock capacity per station
  "budget": 62,                        // total bikes you may place, sum <= budget
  "w_pickup": 1.0,                     // loss weight per lost pickup
  "w_strand": 8.0,                     // loss weight per stranded (permanently lost) bike
  "trips": [[t, o, d, dt], ...]        // the FULL, fixed trip schedule
}
```
`o`, `d` are 0-indexed station ids; `t`, `dt` are non-negative integers.
`trips` is not pre-sorted by `t` — replay in ascending `(t, array index)`
order per the tie-break rule above.

## Answer (stdout JSON)
```json
{"init": [x_1, ..., x_n]}
```
Exactly `n` integers (or floats within `1e-6` of an integer). Each
`0 <= x_i <= capacity[i]`, and `sum(x_i) <= budget`. Any other shape/type, a
non-finite value, an out-of-range station count, or a budget overrun is
rejected outright.

## Objective & scoring
Minimize total loss `= w_pickup * (#lost pickups) + w_strand * (#stranded
bikes)` under the replay described above. Per instance:
`score = min(1, 0.1 * loss(equal_split) / loss(yours))`, where
`equal_split` is the reference allocation that divides the budget evenly
across all stations (clipped to capacity). The final score is the mean
over 10 fixed, seeded instances of varying size, station roles, and budget
tightness — some instances are deliberately harder to test generalization.

## Suggested strategies (increasing sophistication)
- **Equal split** — spread the budget evenly; ignores the trip data.
- **Net-demand water-fill** — rank stations by trips-originated-minus-
  trips-received over the whole day and fill the neediest first; one
  static pass, no replay. Misses stations whose outflow and inflow are
  separated in time rather than in total volume.
- **Marginal-gradient construction** — place bikes one at a time. Before
  each placement, replay the schedule against the CURRENT allocation to
  ask every station "what would your next bike be worth right now?", and
  give it to whichever answer is largest. Querying the real joint system
  at every step already reflects cross-station coupling — a station with
  balanced total traffic but phase-separated in/out flow can show a large
  marginal gain despite ~0 net demand, exactly what a static formula
  misses.
- **Local search polish** — a short round of single-bike swaps against the
  real replay to clean up residual interactions.
