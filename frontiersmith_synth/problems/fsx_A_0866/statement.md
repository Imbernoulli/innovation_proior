# Reserve Dispatch Against a Dead-Time Actuator

A microgrid's frequency deviation `x` (Hz, signed) is driven by a load/generation imbalance
`d[t]` and corrected by dispatched reserve `u[t]`. The reserve actuator (batteries + governors)
has a **deterministic dead-time**: a command issued at step `t` only takes physical effect `L`
steps later. The plant model, honestly:

```
x[0] = x0
x[t+1] = a * x[t] + d[t] - b * u[t-L]      for t = 0 .. T-1     (u[s] := 0 for s < 0)
```

`a` in `(0,1)` is the grid's self-damping; `b > 0` is actuator effectiveness; `L >= 1` is the
dead-time in steps.

**Your program is the dispatch planner.** It is run once, reads one JSON instance from stdin, and
prints one JSON answer to stdout:

```
stdin  (JSON): {"T":.., "L":.., "a":.., "b":.., "c_eff":.., "theta":.., "pen":..,
                "u_max":.., "x0":.., "d_forecast":[T floats]}
stdout (JSON): {"u": [T floats]}          # commanded reserve for t = 0 .. T-1
```

`d_forecast` is a **known day-ahead forecast** of the imbalance. The instance's *actually realized*
imbalance used to grade you differs from `d_forecast` by a small, bounded, unknown-to-you
correction (never revealed) — your dispatch must be genuinely robust, not merely an exact replay
against the forecast. Every `|u[t]| <= u_max`, else that instance scores 0.

**Cost (minimize).** Over the true realized trajectory:

```
cost = c_eff * sum_t u[t]^2  +  sum_{t=1..T} [ x[t]^2 + (pen if |x[t]| > theta else 0) ]
```

`theta` is the protection-relay instability threshold: every step the deviation exceeds it, a
fixed penalty `pen` (given in the input) is added *in addition to* the quadratic tracking term —
a hard cliff, not a smooth penalty, and it is paid **every step** the deviation stays over
`theta`, so a controller that oscillates back and forth across the threshold pays it repeatedly.

**Scoring.** The evaluator also computes, itself, `baseline` = the cost of the simplest sane
dispatch: a constant reserve equal to the forecast's mean (feedforward-only, no feedback at all).
Your instance score is

```
r = clamp( 0.1 + 0.75 * (baseline - cost) / baseline, 0, 1 )
```

so matching the feedforward baseline scores `0.1`; cutting cost to (near) zero approaches `~0.85`;
a dispatch that makes things *worse* than the feedforward baseline (e.g. by tripping the relay
repeatedly) scores `0`. Final score is the mean `r` over 10 fixed instances.

**The trap.** The obvious first design is a proportional/deadbeat controller that reacts to the
*currently visible* deviation `x[t]` — e.g. picking `u[t]` to cancel `x[t+1]` as if the command
landed immediately. Because the actuator dead-time means `u[t]` actually lands at `t+L`, by the
time it acts the system has moved on and any commands already in flight (issued at `t-L+1 .. t-1`,
still pending) get double-counted. When `L` is not small relative to the grid's memory `1/(1-a)`,
or the imbalance changes abruptly, this compounding causes growing oscillation that repeatedly
crosses `theta`, paying the trip penalty over and over — often ending up *worse* than doing almost
nothing.

**What actually works.** Reserve against the **predicted state at the moment the command will
land** (`t+L`), not the visible state now: propagate the current state forward by the known
dynamics over the next `L` steps, using the known forecast for the disturbances in that window and
accounting for the effect of reserve commands already committed but not yet applied, then correct
that predicted value — damping the dead-time itself instead of the symptom you can currently see.

**Feasibility.** `u` must be a length-`T` list of finite numbers, each `|u[t]| <= u_max + 1e-6`.
Any malformed, wrong-length, non-finite, out-of-range, crashing, or timed-out output scores that
instance `0`.

Time limit: 5s per instance. Memory: 512MB. `T` up to 110, so any correct approach finishes in
well under a second.
