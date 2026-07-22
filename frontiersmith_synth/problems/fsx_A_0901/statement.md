# Lazy-Throttle Autopilot

A small autopilot holds a drone's climb rate `y` at a commanded setpoint `sp`
by driving a throttle `u`. The plant is a KNOWN first-order lag:

```
y[t+1] = y[t] + alpha * (u[t] - y[t])
```

with `alpha` given per test case. The autopilot's internal control LAW is
unknown, but its designer left you two things: (1) it is built from the
tracking error `e[t] = sp[t] - y[t]` and a running correction register
`I[t] = I[t-1] + e[t]` (with `I[-1] = 0`) that accumulates as the flight
goes on — this register is `0` at the start of any flight and updates itself
automatically as the aircraft flies; and (2) the throttle is *lazy*: near
its maximum thrust its response compresses smoothly instead of snapping to
a hard stop. Your job: reverse-engineer `u = f(e, I)`.

The catch: the logs you are given were recorded on a **calm** flight —
small setpoint corrections that never push the throttle anywhere near its
limit. In that quasi-linear regime the law looks almost exactly like a
plain linear combination of `e` and `I`. You are graded on a **different,
larger-amplitude** held-out flight for the same aircraft; some grading
flights are only moderately bigger than the calm log, others are outright
**stormy** — large course corrections where the throttle's limited
authority, and the register windup it causes, are fully exposed and an
unclamped linear extrapolation blows up badly.

## Input (stdin)
```
n_rows testId alpha
sp[0]  y[0]  u[0]
sp[1]  y[1]  u[1]
...
sp[n_rows-1] y[n_rows-1] u[n_rows-1]
```
`testId` is the test id. The `n_rows` training rows are noisy calm-flight
readings (setpoint, achieved output, throttle command). The held-out stormy
setpoint profile is a *different, larger-amplitude* sequence for the same
hidden law; it is NOT given to you.

## Output (stdout): one formula
Emit exactly one line:
```
OUT <expr>
```
The grader **rolls your expression forward** over the held-out stormy
setpoints, maintaining `e` and `I` exactly as defined above from your OWN
simulated trajectory (flying it for real, not reading it off a table).

Expressions are arithmetic over `+ - * /`, parentheses, numeric constants,
the unary functions `sig` (logistic), `step` (1 if arg>0 else 0), `relu`,
`tanh`, `absv`, and these variables:

- `e` — current error; `ekJ` — error `J` ticks ago (`J` = `1..8`, e.g. `ek3`).
- `I` — current register; `IkJ` — register `J` ticks ago (e.g. `Ik2`).

The whole expression must use `≤ 40` nodes.

**Illustrative FORM only — NOT the hidden law:**
```
OUT 0.3 + relu ( e - Ik2 ) - 0.4 * sig ( ek1 )
```
This just shows the syntax; the real law is a different shape and you must
discover it from the data.

## Feasibility
The expression must parse under the grammar above (known names/functions
only, finite constants, delays and size within bounds). Any violation, or
any non-finite value produced anywhere during the rollout, scores `0`.

## Objective (minimise)
Let `MSE` be the mean squared error of your rolled-out `y` against the true
held-out trajectory, and `nodes` the number of expression nodes. The grader
forms
```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_plain_P_throttle * (1 + LAMBDA * 3)   # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```
with a small fixed `LAMBDA`, where the baseline is a plain proportional
throttle law fit from your OWN training rows (no memory, no saturation). A
matching fit reproduces `B` (Ratio ≈ 0.1); lowering held-out error raises
the score, with a mild parsimony tax against needlessly large expressions.

## Why the calm log is a trap
On the calm log the throttle never nears its limit, so a linear-in-`(e,I)`
fit looks essentially exact — the compression only shows up as a faint
third-order curvature in the residuals. Extrapolating that linear fit lets
`I` accumulate far past anything the real throttle would ever produce; the
resulting windup sends the rollout off the true trajectory. The faint
curvature is the only clue that a hard ceiling exists at all — and its
scale.

## Constraints
Time limit 5 s, memory 512 MB. `n_rows` is a few hundred rows, each
instance file is well under 1 MB. Scoring is fully deterministic.
