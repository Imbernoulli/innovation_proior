# Highway Sensor Loops Before the Jam

## Problem

A single-lane ring road of length 1 (periodic: position `1` wraps to `0`) carries
traffic whose normalized density `rho(x,t)` in `[0,1]` (`0` = empty, `1` =
bumper-to-bumper) obeys the scalar conservation law

```
rho_t + (f(rho))_x = 0
```

where `f` is a **hidden flux function of density alone** (a "fundamental
diagram") — it does not depend on `x` or `t` directly, only on the local
density value. You are given noisy readings from a sparse ring of inductive
**sensor loops**, recorded on a calm day: over the whole recorded window the
density stays smooth and no jam (shock) has formed.

Your job: submit a closed-form expression for `f(rho)`. The grader feeds your
`f` into a **fixed, provided numerical scheme** (Lax–Friedrichs, entropy
respecting) and evolves a *different*, held-out, *steep* initial density
profile on the same road — one that DOES break into a genuine traffic-jam
shockwave — then compares your simulated density against the true one at
several later times.

## Input (stdin)

```
M K t
t_0 x_0 rho_0
t_0 x_1 rho_1
...
t_{K-1} x_{M-1} rho_{MK-1}
```

`M` sensor-loop positions `x_m = m/M` (fixed, equally spaced around the
ring), `K` observation times `t_k` (`t_0 = 0`), `t` is the test id. Each row
is one noisy reading `rho(x_m, t_k)`; all readings come from the same smooth,
pre-jam evolution.

## Output (stdout)

One line: a single Python expression over the variable `rho`, using
`+ - * / **`, parentheses, numeric constants, and the unary functions
`exp log sin cos sqrt tanh abs`.

**Illustrative FORM only — NOT the hidden law** (the real shape must be
discovered from the data):

```
0.5*sin(3*rho) + 0.2*rho
```

## Feasibility

Your expression must parse under the grammar above (only `rho` and the
listed functions, finite constants, `<=40` AST nodes) and evaluate to a
finite real number everywhere the reference solver visits during the
held-out rollout. Any violation scores `0`.

## Objective (minimize)

Let `MAE` be the mean absolute error between your flux's simulated density
profile and the true density profile on the held-out steep initial
condition, averaged over several later checkpoint times, and let `nodes` be
your expression's AST node count. The grader forms

```
F = MAE * (1 + LAMBDA*nodes)
B = MAE_of_zero_flux * (1 + LAMBDA*1)     # internal baseline: "no traffic moves"
Ratio = min(1000, 100*B/F) / 1000
```

with a small fixed `LAMBDA`. Submitting `f(rho) = 0` reproduces `B` exactly
(`Ratio ~= 0.1`). Lower held-out error raises the score; a small parsimony
tax discourages needlessly baroque expressions.

## Why the calm day is a trap

On the calm-day data the road rarely gets dense enough, or its gradients
steep enough, for the SHAPE of `f` near high density to matter much — many
different flux functions fit the smooth training snapshots almost equally
well. But the shockwave's LOCATION and SPEED on the held-out jam are governed
by the exact values of `f` at the low and high densities on either side of
the jam (the Rankine–Hugoniot relation
`speed = (f(rho_R) - f(rho_L)) / (rho_R - rho_L)`), and by whether the
recovered `f` even makes characteristics converge there. A model that only
mimics the observed PROFILES — e.g. treating the pattern as a rigidly
translating shape and fitting one bulk speed — reproduces the training
snapshots but has no way to predict where or how fast a jam will form.
Recovering the flux LAW itself from how individual density LEVELS move (the
conserved quantity's own characteristics, visible even in smooth data)
generalizes to the shock for free.

## Constraints

Time limit 5s, memory 512MB. `M` is a few dozen, `K` is 6. Scoring is fully
deterministic; the reference solver and the held-out initial condition are
fixed and regenerated identically by the grader from the test id `t`.
