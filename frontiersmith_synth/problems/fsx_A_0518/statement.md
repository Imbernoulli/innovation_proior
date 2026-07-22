# Antique Greenhouse Thermostat

You are restoring an antique greenhouse whose bimetal thermostat drives a heater
through a mechanism nobody documented. You have a data logger that records, at
each tick, the **drive temperature** `d[t]` (a normalised sensor reading in
`[0,1]`) and the **heater output** `y[t]`. Your job: reverse-engineer a predictor
of `y` from `d`.

The catch: the logs you are given were recorded on a calm day, under a **slow,
monotone-ish** drive. In that quasi-static regime the mechanism looks almost like
a plain static curve `y = f(d)`. But you will be graded on a **fast, gusty**
day — a held-out drive that reverses direction often and quickly — where any
hidden state in the mechanism is fully exposed. Predictors that merely memorise
the calm-day curve extrapolate badly.

## Input (stdin)

```
n t
d[0]  y[0]
d[1]  y[1]
...
d[n-1] y[n-1]
```

`t` is the test id; `n` training rows follow, each a drive value and heater
value (floats). The held-out grading trace is a **different, faster** drive for
the same mechanism; it is NOT given to you.

## Output (stdout): a stateful predictor in a tiny DSL

Emit at most two statements:

```
LATCH <set_expr> | <reset_expr>      (optional; at most one)
OUT   <expr>                          (required; the emitted heater value)
```

The grader **rolls your program forward** over the held-out drive, carrying one
latch register `S` across ticks:

- Each tick it evaluates `set_expr` and `reset_expr`. If `set_expr > 0` the latch
  is set to `1`; else if `reset_expr > 0` it is reset to `0`; **else it HOLDS** its
  previous value. (No `LATCH` line ⇒ `S` stays `0`.)
- Then it evaluates `OUT <expr>` to produce `y_hat[t]`.

Expressions are arithmetic over `+ - * /`, parentheses, numeric constants, the
unary functions `sig` (logistic), `step` (1 if arg>0 else 0), `relu`, `tanh`,
`absv`, and these variables:

- `d` — the current drive; `dkJ` — the drive `J` ticks ago (e.g. `dk3`).
- `S` (=`S0`) — the current latch; `SkJ` — the latch `J` ticks ago (e.g. `Sk2`).

`LATCH` conditions may reference the drive taps **only** (never `S`/`SkJ`).
Delays `J` must be `1..24`; the whole program must be `≤ 80` nodes.

**Illustrative FORM only — NOT the hidden mechanism:**

```
LATCH 0.5 - dk1 | tanh ( d ) - 0.3
OUT   0.1 + relu ( d - dk2 ) + 0.4 * S
```

This just shows the syntax; the real mechanism is different and you must
discover its shape from the data.

## Feasibility

The program must parse under the grammar above (known names/functions only,
finite constants, delays and size within bounds). Any violation, or any
non-finite value produced during the rollout, scores `0`.

## Objective (minimise)

Let `MSE` be the mean squared error of your rolled-out `y_hat` against the true
held-out heater trace, and `nodes` the number of expression nodes in your program.
The grader forms

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_constant_0.5 * (1 + LAMBDA * 1)      # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. A constant predictor reproduces `B` (Ratio ≈ 0.1);
lowering held-out error raises the score, but a parsimony tax discourages
needlessly large programs. Report the highest Ratio you can.

## Why the calm-day curve is a trap

On the slow branch, a single value of `d` maps to essentially one `y`, so a
memoryless `sig`/threshold fit looks excellent. On the fast held-out drive the
drive re-crosses the same band from BOTH directions, and the heater lags — so
the residuals of any memoryless fit line up with the **direction of change**, not
the value of `d`. That is the signature of hidden state you must model.

## Constraints

Time limit 5 s, memory 512 MB. `n` is a few hundred rows. Scoring is fully
deterministic.
