# Two-Speed Reservoir: Dam Release Scheduling Against Seeded Rainfall

## Story

You operate a dam. A `T`-day rainfall record has already been observed by the
weather service and handed to you in full — you know every day's inflow in
advance. Each day you choose how much water to release downstream. Get it
wrong and you pay one of two penalties: let the reservoir overtop and you pay
a **flood penalty** on the spilled volume; let it run dry and you pay a
**shortage penalty** on the deficit below the safe operating floor. Your goal
is to minimize the total penalty over the whole record.

The catch: the spillway/outlet works can only pass **`Rmax`** units of water
per day, no matter how much you ask for. The rainfall record itself mixes two
speeds: a slow, smooth seasonal drift (rises and falls gently over the whole
record) with occasional fast, large storm bursts riding on top of it, lasting
a day or two each. Both are baked into one plain sequence of numbers — nothing
in the input tells you which day is "storm" and which is "seasonal"; you only
see the combined inflow. If the reservoir is sitting near capacity when a big
storm lands, `Rmax` is not enough to drain it away in a single day and the
excess must spill — the only way to avoid that is to have already lowered the
level *before* the storm arrives.

## Input (public instance, one JSON object on stdin)

```json
{"name": "s301_singlestorm", "T": 60, "cap": 1000.0, "Rmax": 40.0, "L0": 500.0,
 "min_level": 150.0, "flood_coef": 6.0, "shortage_coef": 3.0,
 "inflow": [13.9, 15.2, ..., 707.4, 245.9, ...]}
```

- `T` (int): number of days.
- `cap` (float): reservoir capacity.
- `Rmax` (float): maximum release allowed on any single day.
- `L0` (float): starting level, `0 <= L0 <= cap`.
- `min_level` (float): safe operating floor; ending a day below it costs a
  shortage penalty.
- `flood_coef`, `shortage_coef` (float): the two penalty weights, given
  per-instance — read them and trade off accordingly.
- `inflow` (list of `T` floats): day-by-day inflow, `>= 0`.

## Output (one JSON object on stdout)

```json
{"releases": [12.0, 12.0, 40.0, ...]}
```

Exactly `T` numbers, each finite and in `[0, Rmax]` — the amount you *ask* to
release on each day. Wrong length, a non-numeric/out-of-range/non-finite
entry, a crash, a timeout, or non-JSON output scores that instance `0.0`.

## Dynamics and scoring (deterministic)

Starting from `L = L0`, for each day `t = 0..T-1` in order:

```
avail          = L + inflow[t]
actual_release = clip(releases[t], 0, Rmax, avail)   # can't release water you don't have
raw            = avail - actual_release
if raw > cap:
    penalty += flood_coef * (raw - cap)               # flood: spill the excess
    L = cap
else:
    L = raw
    if L < min_level:
        penalty += shortage_coef * (min_level - L)     # shortage: below the safe floor
```

`penalty` accumulates over all `T` days; lower is better. The evaluator also
computes, for the SAME instance, `y_base` = the penalty of doing nothing
(releasing `0` every day, the weak reference) and normalizes:

```
r = clamp(0.1 + 0.9 * (y_base - your_penalty) / (1.2 * y_base), 0, 1)
```

Matching the do-nothing baseline scores about `0.1`; doing worse scores `0`;
beating it scores higher, but the ceiling sits above the physically-optimal
zero-penalty outcome (an unreachable reference), so even a flawless run does
not score `1.0` — there is always room to be more efficient. Your final score
is the mean of `r` over 10 instances with varied capacity, release limits,
storm timing/size, and some tight-budget / held-out cases.

## Notes

- Everything you need (`inflow`, `Rmax`, coefficients, floor) is in the
  public instance; nothing about scoring is hidden except the internal
  construction of `y_base`.
- Scoring never measures wall-clock time; treat the per-instance limit as a
  compute budget.
- Your program runs in an isolated subprocess and sees only the public
  instance above.
