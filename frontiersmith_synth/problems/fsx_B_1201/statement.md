# The Temperature Curve That Suddenly Turns Up

## Problem

A self-heating device (a battery pack, a chemical reactor, a densely-packed
electronics module) sits in an enclosure held at ambient temperature `Ta`. Two
competing processes set its steady operating temperature `T`:

- **Self-heating** grows with temperature itself, `G(T) = A * exp(b*T)` (an
  Arrhenius-type kinetics — hotter means it generates heat faster).
- **Cooling** removes heat proportionally to the excess above ambient,
  `h * (T - Ta)`, but the cooling hardware has a hard **capacity limit**: it
  can never remove more than `Hmax` units of heat per unit time, however hot
  the device gets.

At steady state, generation equals removal. Below a **critical ambient
temperature**, this balance has a solution the device settles into and stays
at. Above it, self-heating outpaces every ounce of cooling the hardware can
supply — even running flat out at `Hmax` — and no steady state exists: the
device does not sit anywhere near the smooth trend of the sub-critical data.
It runs away until a protection circuit trips at a fixed cutoff temperature
`Tfail` (given to you).

You are given `b`, `h`, `Hmax`, `Tfail`, and steady-state measurements
`(Ta_i, T_i)` taken only at **sub-critical** ambients (the enclosure was never
deliberately pushed into runaway while logging — that destroys hardware). `A`
is not given: back it out from the data yourself. Your job is to predict `T`
at held-out ambient temperatures, some of which lie **above** the (unknown to
you) critical value.

## Input (stdin)

```
n t
b h Hmax Tfail
Ta_0 T_0
Ta_1 T_1
...
Ta_{n-1} T_{n-1}
```

`t` is the test id. All temperatures use one consistent, unitless instrument
scale (not necessarily Celsius). Each `T_i` includes small measurement noise.

## Output (stdout): a threshold predictor, exactly 3 lines

```
THRESH <expr>
BELOW  <expr(Ta)>
ABOVE  <expr(Ta)>
```

`THRESH` is a **constant** expression (must not reference `Ta`) giving your
estimate of the critical ambient temperature. For a query ambient `q`: if
`q < THRESH` your prediction is `BELOW` evaluated at `Ta=q`; otherwise it is
`ABOVE` evaluated at `Ta=q`. Expressions use `+ - * / **`, unary `-`,
parentheses, numeric constants, the variable `Ta` (forbidden in `THRESH`), and
the unary functions `sqrt log exp sig tanh absv`. At most 40 AST nodes per
expression.

**Illustrative FORM only — NOT the hidden law:**

```
THRESH 41.0
BELOW  3.0 + 0.4*Ta - 0.01*Ta**2
ABOVE  70.0
```

## Feasibility

Output must parse under the grammar above (known names/functions, finite
constants, node budget). Any parse error, disallowed name, oversized
expression, or non-finite value produced anywhere while grading scores `0`.

## Objective / Scoring (maximize)

The grader regenerates a **held-out** grid of ambients (same `t`, fully
deterministic) spanning three regimes never shown together in training:
interpolation inside the training range, **near-critical** ambients just
below the true threshold, and **super-critical** ambients above it (true
value `Tfail`). For each held-out point with true value `T_i` and your
prediction `p_i`:

```
d_i    = min(1, |p_i - T_i| / (|p_i| + |T_i| + 1e-6))
metric = mean_i d_i
O      = metric * (1 + LAMBDA * total_nodes)
B      = baseline_metric * (1 + LAMBDA * 1)   # baseline: constant = median(train T)
Ratio  = min(1000, 100 * B / O) / 1000
```

`total_nodes` is the summed AST node count of all three expressions. A
constant predictor reproduces the baseline (`Ratio ~ 0.1`). Measurement noise
on the held-out grid keeps even a strong model below `1.0`.

## Why the smooth fit is a trap

A curve fit through only sub-critical `(Ta,T)` pairs — even a very good one —
keeps extrapolating the same *shape* past the last training point: a bit more
of the same smooth climb. It has no way to know the cooling hardware runs out
of capacity. The real system does not climb smoothly past the threshold — it
jumps straight to `Tfail`. Spotting the threshold means comparing the *rate*
heat is generated against the *rate* it can be removed near the edge of the
observed range, not extrapolating the temperature curve itself: since
`G(T) = A*exp(b*T)` and `h`, `Hmax` are given, the row-wise self-heating level
`A` — recoverable from each training pair via
`A = h*(T_i - Ta_i)*exp(-b*T_i)` (an exact identity at steady state) — tells
you precisely the ambient at which generation first reaches the cooling
ceiling `Hmax`. That crossing is where the curve turns up for good, and it is
usually well past the last training row.

## Constraints

Time limit 5 s, memory 512 MB. `n` is at most a few dozen rows. Scoring is
fully deterministic.
