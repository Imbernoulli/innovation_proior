# Charge the Fleet, Not the Van: Slot Allocation for Electric Delivery Vans

## Story

A depot runs a fleet of electric delivery vans. Each van has a fixed multi-stop
route it must drive in order. Every leg of a route costs a different amount of
battery energy depending on terrain (flat, hilly, mountain legs draw different
amounts per minute of travel), so how much energy a van has burned by any point in
its route depends on the actual legs it has driven, not just elapsed time.

Every route passes through exactly two charge-capable stops: an **early** one and
a **late** one. Charging to full at *either one alone* is always enough energy to
finish the whole route (topping up at the early stop covers the rest of the route
by itself; so does topping up at the late stop). A van never strands from running
low, but if it charges *less* than it actually needs, it can still run out of
energy on a later leg and forfeit every remaining stop on its route.

Chargers are shared, scarce infrastructure: each has only a handful of parallel
plugs. Many vans' late stops sit on the **same popular charger hub**. If every van
plans for itself and waits until its late stop to charge (the natural,
individually-correct choice — the early stop looks unnecessary to a lone van),
their charging windows cluster and they queue for a free plug. Waiting doesn't
drain the battery, but it eats delivery slack, and later stops slip past deadline.
A fleet-level plan does better by having *some* vans charge at their early stop
instead — before they strictly need to — spreading demand across two chargers and
two points in time.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "fleet18_trap",
  "chargers": [ {"id": 0, "slots": 2, "rate": 6}, {"id": 1, "slots": 1, "rate": 6}, {"id": 2, "slots": 2, "rate": 5} ],
  "vans": [
    { "id": 0, "p1": 2, "p2": 5, "p1_charger": 2, "p2_charger": 0,
      "capacity": 34, "deadlines": [9, 20, 30, 42, 55, 70, 84],
      "legs": [ {"time": 9, "energy": 9}, "... one per route leg ..." ] },
    "... more vans ..."
  ]
}
```

- `chargers`: each has an `id`, a plug count `slots`, a charge `rate` (energy
  units/minute).
- Each van has `legs` (one per route leg, in order, each with `time` minutes and
  `energy` consumed), `p1`/`p2` (1-indexed stop numbers of its early/late
  charge-capable stops, `1 <= p1 < p2 < len(legs)`), `p1_charger`/`p2_charger`
  (charger `id` at each), `capacity`, and `deadlines` (one per stop `1..len(legs)`,
  minutes from t=0).
- A van starts full at stop 0 and drives its legs in order. Charging at a stop (if
  requested) happens *after* that stop's delivery and delays only *later* stops.

## Output (one JSON object on stdout)

```json
{"vans": [{"id": 0, "charge_at_p1": 0, "charge_at_p2": 22}, ...]}
```

- Exactly one entry per van `id` that appears in the input, each with non-negative,
  finite `charge_at_p1` and `charge_at_p2` (energy units to request there; 0 means
  skip). Requests are clamped to whatever battery headroom the van actually has
  when it arrives — asking for more than needed is never penalized, only wasted.
- Missing/duplicate/extra van ids, wrong types, or negative/non-finite amounts make
  the WHOLE instance score `0.0`, as does a crash, timeout, or non-JSON output.

## Objective and scoring (deterministic)

For each instance the evaluator runs the same discrete-event fleet simulator (all
vans and all chargers together, plugs assigned FIFO by real arrival order) on your
plan and on its own **floor plan**: never charge at all. Every route needs at
least one recharge to finish (no single leg or route half alone exceeds a van's
capacity, but the whole route does), so never charging reliably strands a van
partway through. Let `frac_floor` and `frac_cand` be the fraction of (van, stop)
deliveries made **on time** (arrival at or before that stop's deadline; a stranded
van forfeits every stop from the point it strands onward) under each plan. Then:

```
r = clamp( 0.1 + 0.9 * (frac_cand - frac_floor) / max(0.12, 1.0 - frac_floor), 0, 1 )
```

Matching the do-nothing floor scores about `0.1`. Delivering more stops on time
scores higher, up to `1.0` (needs every stop of every van on time — usually
unreachable once contention is real, leaving headroom). Doing worse than the
floor scores below `0.1`, down to `0.0`. Your final score is the mean of `r` over
10 instances of varying fleet size and charger scarcity, including harder
held-out fleets.

## Notes

- Scoring never measures wall-clock time; treat the time limit as a compute budget.
- Your program runs in an isolated subprocess and sees only the public instance.
