# Beads on a Wire: Loop-Line Holding Control

`N` buses share a circular loop line. Passengers build up at every stop at a
steady rate, so a bus's **dwell time** grows *linearly* with the headway since
the **preceding** bus visited: `dwell = beta * headway`. A bus that falls
slightly behind picks up more passengers, dwells longer, and falls further
behind — a positive feedback loop that, left alone, drifts buses into clumps
("bunching"), which is bad for waiting passengers: squared wait time grows
with how uneven the gaps become.

The operator's only lever is **holding**: a dispatcher may make a bus wait
*extra* time at a stop before departing — it can only ADD delay, never
subtract it. Because the line is a **loop**, gaps wrap around: bus `i`'s
headway is measured against its predecessor, and holding bus `i` doesn't just
change its own future gap — it also reshapes the gap left for the bus
immediately **behind** it. The fleet is a *ring of coupled oscillators*: a
perturbation on one bus propagates around the ring, and some ring-wide
oscillation patterns are far harder to damp than others.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (a fixed holding-control law) to **stdout**.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide a control law ...
print(json.dumps({"gain_back": gb, "gain_fwd": gf, "target_frac": tf, "cap_frac": cf}))
```

### Public instance (stdin)

```json
{
  "name": "loopC",
  "n_buses": 8,             // N buses on the loop
  "beta": 0.05,             // dwell-feedback gain (dwell = beta * own headway)
  "rounds": 12,             // K, number of stop-visits simulated
  "nominal_headway": 50.0,  // Hnom, the schedule's target spacing
  "max_hold": 17.5,         // hard cap on hold time per bus per stop
  "capacity_weight": 0.15,  // gamma, weight on total hold time in the score
  "initial_headways": [ ... N floats, sum == N * nominal_headway ... ]
}
```

### Answer (stdout) — a control law, applied for the WHOLE horizon

```json
{ "gain_back": 0.7, "gain_fwd": 0.3, "target_frac": 1.0, "cap_frac": 0.9 }
```

Each field must be a finite number: `gain_back, gain_fwd` in `[-3, 3]`,
`target_frac` in `[0.5, 1.5]`, `cap_frac` in `[0, 1]`. Any missing field,
wrong type, out-of-range or non-finite value, a crash, a timeout, or non-JSON
output makes that instance score `0.0`.

## Dynamics (computed by the evaluator, exactly this formula)

Let `H_i(k)` be bus `i`'s headway (gap since its predecessor) at round `k`,
`Ht = target_frac * nominal_headway`. The evaluator applies your law at every
round, to every bus:

```
u_i(k)     = clip( gain_back*(Ht - H_i(k)) + gain_fwd*(Ht - H_{(i+1)%N}(k)), 0, cap_frac * max_hold )
H_i(k+1)   = (1+beta)*H_i(k) - beta*H_{(i-1)%N}(k) + u_i(k) - u_{(i-1)%N}(k)
```

`H_{(i+1)%N}(k)` is the CURRENT headway of the bus immediately behind bus `i`
— the gap bus `i` is about to leave in its wake. A law using only `H_i(k)`
(`gain_fwd=0`) reacts to your own lateness alone, like chasing a fixed
personal schedule. Also using `H_{(i+1)%N}(k)` reacts to what your holding
does to the bus behind you — the ring as coupled, not as N independent buses.

## Scoring (deterministic)

For each instance:

```
wait_term = mean over k in [0,rounds), i in [0,N)  of  (H_i(k) - nominal_headway)^2
hold_term = mean over k in [0,rounds), i in [0,N)  of  u_i(k)
obj       = wait_term + capacity_weight * hold_term        # MINIMIZE
q_base    = obj of the internal NEVER-HOLD law (gain_back=gain_fwd=0)
ratio     = max(0, obj / q_base)
r         = clamp( 0.1 + 0.9 * (1 - sqrt(ratio)), 0, 1 )
```

(`q_base` swings over orders of magnitude across instances since the
instability is exponential in the horizon, so the raw gap is compressed with
a square root before the affine map — this keeps the score sensitive to
controller quality instead of letting every stabilizing policy round up to
the same near-perfect number.)

Matching never-hold scores `~0.1`; a perfectly flat, zero-cost spacing (an
unreachable ideal, since some transient wait is unavoidable and holds can't
be negative) would score `1.0`. **Ratio** is the mean of `r` over 10 seeded
instances (varying `N`, `beta`, horizon, hold cap, capacity weight, and the
shape of the initial perturbation — alternating, single-spike, or
mild-random); several are larger / harder held-out cases. **Vector** lists
the per-instance scores.

## Suggested strategies

1. **Never hold** — dwell-feedback runs free; guaranteed to drift.
2. **Target-schedule chasing** — hold each bus toward the nominal headway
   using only its own gap (`gain_fwd=0`), one fixed gain on every instance
   regardless of that instance's actual feedback strength.
3. **Search a two-way law** — the dynamics are stated in full; replicate them
   internally and search `(gain_back, gain_fwd, target_frac, cap_frac)` to
   trade correction speed against capacity spent holding, per instance.
4. **Reason about the ring's oscillation modes** — an all-shift-together
   perturbation is harmless; an alternating, every-other-bus perturbation is
   the hardest pattern to damp, and the one a backward-only law struggles
   most to control without overcorrecting or overspending the hold budget.
