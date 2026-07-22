# Boiler Startup: Step-Size Control for a Relaxation Trajectory with a Wandering Stiff Transient

A controlled process variable `y(t)` (think a boiler drum level, or a reactor
temperature deviation) obeys a scalar relaxation law

```
dy/dt = f(t, y) = -K(t) * (y - E(t))
```

`E(t)` is a smooth, externally-driven setpoint (a slow drift plus one
sinusoidal disturbance — mild curvature everywhere). `K(t)` is the plant's own
relaxation rate, given **piecewise-constant** over `[0, T]`: mostly a small
`K_base`, but with one or two short windows where the rate jumps to several
hundred — the plant is driven hard for a while, then returns to gentle
tracking. These stiff windows can sit anywhere along the path (early, or
later, or both) — read `K_segments` and act on it.

You must report the trajectory's value at a set of fixed **checkpoint** times,
but you never report values directly. Instead you submit a **step schedule**:
a strictly increasing sequence of times partitioning `[0, T]`, and for each
step, which update rule to use:

- **`explicit`** — one classical RK4 step. 4th-order accurate, but only
  *conditionally stable*: once `h * K(t)` exceeds roughly `2.5` anywhere
  inside the step, the step genuinely diverges numerically.
- **`implicit`** — one backward-Euler step, with `K`,`E` evaluated at the
  step's *end* time. Unconditionally stable for any `h` (since `K >= 0`), but
  only 1st-order accurate — wasteful if used where accuracy, not stability,
  is the limiting factor.

Every step costs a fixed number of **derivative-evaluation units** against a
hard budget `max_evals`: `cost_explicit` units per explicit step,
`cost_implicit` per implicit step (`cost_implicit > cost_explicit`, the extra
work of a stabilized solve). The evaluator — not you — performs the actual
integration from your schedule, so there is no way to submit a cheap-looking
schedule while secretly computing a more accurate one elsewhere: your
schedule *is* the computation that gets scored.

**Feasibility** (violating any of these scores `0` on that instance):
- Step times strictly increase from `> 0`, the last step's `t1` equals `T`.
- Every time in `checkpoints` must appear **exactly** as some step's `t1`
  (you may freely add extra step boundaries anywhere in between, e.g. to
  cross a stiff window safely, without those extra points being checkpoints).
- Total spent evaluation units (summed step costs) must not exceed
  `max_evals`.

**Objective (minimize).** The evaluator re-integrates your exact schedule and
compares the state at each checkpoint against a high-accuracy reference,
reporting the RMSE (capped at a fixed ceiling before scoring, so a few
catastrophically diverged checkpoints don't swamp an otherwise-good
schedule). There is no easy optimum: tiny explicit steps everywhere waste
budget on the smooth part; a uniform step coarse enough to afford the smooth
part will *diverge* inside a stiff window; and always using the stable
implicit rule is only 1st-order accurate, wasteful once you're back in a
gently-curving region.

## Public instance (stdin JSON)

```json
{
  "T": 15.62,
  "y0": 7.83,
  "K_segments": [
    {"t0": 0.0,  "t1": 0.41, "K": 0.62},
    {"t0": 0.41, "t1": 1.90, "K": 240.7},
    {"t0": 1.90, "t1": 15.62, "K": 0.62}
  ],
  "E_coef": {"e0": 1.1, "e1": 0.06, "e2": 1.8, "w": 0.9, "phase": 2.3},
  "checkpoints": [1.12, 2.23, 3.35, "...", 15.62],
  "max_evals": 472,
  "cost_explicit": 4,
  "cost_implicit": 8
}
```

`E(t) = e0 + e1*t + e2*sin(w*t + phase)`. `K_segments` covers `[0, T]`
contiguously and in order.

## Answer (stdout JSON)

```json
{"steps": [{"t1": 0.02, "method": "implicit"}, "...", {"t1": 15.62, "method": "explicit"}]}
```

## Scoring

The evaluator computes a baseline `b` from its own "trivial" construction: a
**uniform**, non-adaptive grid of explicit RK4 steps spending (almost) the
full `max_evals` budget, still forced to land exactly on every checkpoint.
This is always feasible by construction. For a feasible answer with
(capped) objective `obj`:

```
r = min(1, 0.1 * b / obj)
```

so matching the uniform baseline maps to exactly `0.1`, and a schedule with
`k` times lower error maps to `min(1, 0.1*k)`. The reported `Ratio` is the
mean of `r` over 10 deterministic, seeded instances (some with a generous
budget, some deliberately tight around the stiff window). Infeasible or
malformed answers score `0` on that instance.

Your program reads one public instance JSON from stdin and writes one answer
JSON to stdout. It runs in an **isolated subprocess** and only ever sees the
public instance.
