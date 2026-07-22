# Curb Reservations First (Format B, isolated)

A small pooled-shuttle fleet serves an **online** stream of ride requests on
a 1-D strip of `S` curb slots (`0..S-1`). Every curb slot can service at
most `curb_cap` vehicles in the **same tick**; a vehicle that reaches a
curb already at capacity that tick must **circle** (a fixed penalty,
retried next tick). Each vehicle can carry up to `veh_capacity` riders at
once — picking up a second rider before dropping the first (**pooling**)
is allowed, but the vehicle's real path is replayed, so any zig-zag it
causes is charged as **detour** to every rider onboard while it happens.

Each request has a **preferred** pickup curb but tolerates boarding one
slot to either side for a small walk penalty — the physical curb actually
used for a pickup is a real choice your program makes, not fixed data.

## How you are invoked
The episode runs `T` ticks. Your program is invoked **once per tick** (a
fresh, isolated process each time) and sees every request released so far
that is still unassigned, plus every vehicle's status/position. You return
route assignments for any subset of the **currently idle** vehicles only;
anything you don't assign simply stays pending for a later tick.

### Input (stdin JSON, one call per tick)
```json
{
  "S": 10, "V": 3, "veh_capacity": 2, "T": 14, "curb_cap": 1, "tick": 3,
  "vehicles": [{"id": 0, "pos": 4, "status": "idle"}, ...],
  "pending": [
    {"id": 7, "release_tick": 2, "pickup_pref": 5,
     "pickup_options": [4, 5, 6], "dropoff": 8}, ...
  ]
}
```

### Output (stdout JSON)
```json
{"assign": {
  "0": [{"action": "pickup", "request": 7, "curb": 5},
        {"action": "pickup", "request": 9, "curb": 5},
        {"action": "dropoff", "request": 9},
        {"action": "dropoff", "request": 7}]
}}
```
Keys are idle vehicle ids (as strings). Each route is an ordered stop list:
a `pickup` stop MUST give a `curb` from that request's `pickup_options`; a
`dropoff` stop needs only `request` (its curb is fixed). Onboard riders
must never exceed `veh_capacity`, and every picked-up request in a route
must be dropped off later in the SAME route. Any malformed/out-of-range
assignment, an unavailable vehicle, or a request not currently pending
fails the WHOLE test case (score 0 for it).

## Physics the grader replays
Each tick, every vehicle with a pending route moves at most 1 slot toward
its next stop's curb; if that lands it exactly on the curb (now or
earlier), it attempts to **service** the stop THAT SAME tick. If more
vehicles are ready at the SAME curb the SAME tick than `curb_cap` allows,
lowest-vehicle-id wins; the rest pay the circle penalty and retry next
tick. A pickup costs `walk_penalty * |chosen_curb - preferred_curb|` and
starts the rider's wait; a dropoff costs that wait plus the rider's full
realized ride time in ticks — never less than the direct pickup-to-dropoff
distance (pooling/queueing can only add ticks, never subtract, so any
excess above direct is the real detour cost pooling caused this rider). An
un-picked-up request pays a large abandonment penalty at episode end.

## Objective & scoring
**Minimize** total cost (wait + ride-time + walk + circle + abandon,
summed over all 10 fixed seeded cases). Per case the grader computes two
references from the hidden data: `W` = the cost of a deliberately weak
single-vehicle-serial baseline, and `O` = the unattainable oracle sum of
direct pickup-to-dropoff distances (a true lower bound, since ride-time
always dominates it). Your score is `clip((W - your_total) / (W - O), 0,
1)`; the final Ratio is the mean over all 10 cases.

## Why nearest-vehicle dispatch is a trap
The obvious policy: every tick, send the nearest idle vehicle straight at
each request's preferred curb, one rider per trip. Some cases plant
several requests at the **same** preferred curb with vehicles
**equidistant** from it — nearest-vehicle dispatch sends them all
converging on that curb the same tick (guaranteed circling), and never
pools riders it could combine.

## Suggested strategies
- **Blind throughput-limited** — one dispatch per tick, direct routes, no
  pooling, no curb choice.
- **Nearest-vehicle instant dispatch** — greedy matching every tick,
  always the preferred curb, never pools.
- **Curb-reservation-first** — group pending requests by pickup location,
  pool up to `veh_capacity` per shared trip, spread the physical curb used
  across each group's tolerated options so two pickup events don't land on
  the same curb the same tick, and only THEN match the nearest available
  vehicle to each resolved reservation.
