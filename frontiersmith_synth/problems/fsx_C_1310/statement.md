# Pointing a Telescope That Cannot Be Everywhere

## Story

An Earth-observation satellite crosses a strip of ground targets twice: an early
**pass 1** and a later **pass 2**, separated by an orbital gap. Before each pass the
telescope points at position `0`. Slewing from position `a` to `b` costs
`|a-b| * slew_rate + settle` seconds; imaging a target then costs its `dwell`
seconds. Each pass has a hard time budget -- once the running clock would exceed
it, that pass simply stops taking new images (later ids in your plan for that pass
are skipped, not an error).

A target's collected value **decays** with the absolute mission clock at the
moment it is imaged: `value * max(0, 1 - decay_rate * mission_time)`, floored at 0.
Mission time in pass 1 is just the pass-1 clock; in pass 2 it is
`pass_gap + pass-2 clock`, so a pass-2 image of the same target is always worth
less than an equally-timed pass-1 image -- whatever the target shows is evolving.

Every target also carries a per-pass **cloud forecast**: `cloud_forecast_p1` /
`cloud_forecast_p2` is the probability it is cloud-covered during that pass. You
only ever see the forecast. Whether it is ACTUALLY clouded during the pass in which
you image it is hidden: if it is, that observation returns **zero** value (the time
is still spent, burned for nothing). A target can be imaged in **at most one** pass
total.

Imaging the richest target next ignores two costs: how far away (hence expensive)
it is to reach, and how likely it is to be clouded out right now versus later.
Your job is to route each pass and decide, per target, whether to risk pass 1 or
bank on pass 2, to **maximize total collected value**.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "orbit04_cloudtrap", "slew_rate": 1.0, "settle": 3.0,
  "pass1_budget": 140.0, "pass2_budget": 140.0, "pass_gap": 380.0,
  "targets": [
    {"id": 0, "x": 40.0, "value": 240, "dwell": 6.0, "decay_rate": 0.0004,
     "cloud_forecast_p1": 0.88, "cloud_forecast_p2": 0.10},
    ...
  ]
}
```

`x` is the target's pointing position. IDs are `0..len(targets)-1`.

## Output (one JSON object on stdout)

```json
{"pass1": [id, id, ...], "pass2": [id, id, ...]}
```

Each list is a **visiting order** for that pass, starting from position 0. A
target id may appear in **at most one** of the two lists (never both, never
twice); a list may be any length, including empty, and may include ids that end
up not fitting the budget (simply skipped, not invalid). Any id out of range, a
duplicate (within a list or across both), a crash, a timeout, or non-JSON output
makes the whole instance score `0.0`.

## Objective and scoring (deterministic)

Per instance the evaluator also computes, itself, with full information:

- `y_base` = the value-sorted-descending, pass-1-ONLY, first-fit-in-that-order
  plan -- exactly "image the highest-value target next," ignoring slew cost,
  cloud forecast, decay, and pass 2 entirely. This is the weak reference.
- `y_ub` = the sum of every target's raw `value`, ignoring cost/decay/cloud -- a
  loose, generally unreachable upper bound.
- `y_cand` = your plan's simulated collected value (cloud + decay applied
  honestly, per the rules above).

```
r = clamp( 0.1 + 0.9 * (y_cand - y_base) / max(1e-9, y_ub - y_base), 0, 1 )
```

Matching the weak reference scores about `0.1`; doing worse scores below `0.1`;
collecting more value scores higher, capped at `1.0` (the loose upper bound keeps
even strong plans well below `1.0`). Your final score is the mean of `r` over 10
instances of varying size, budget, and risk profile, including harder held-out
orbits.

## Notes

- Scoring never measures wall-clock time; treat the budgets as an operations
  constraint to route within, not a speed contest.
- Your program runs in an isolated subprocess and sees only the public instance
  above -- never the hidden actual cloud state.
