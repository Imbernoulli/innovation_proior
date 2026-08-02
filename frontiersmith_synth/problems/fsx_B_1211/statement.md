# When the Shelf Lets Go

A field team is forecasting calving on a floating ice shelf. Along the calving
front they log several **segments**. Each segment has a thickness `H(t)` that
thins slowly and linearly, `H(t) = H0 - gamma*t`, and a rift whose depth
`D(t)` grows because of ongoing **stress accumulation** driven by the local
strain rate:

- **Phase 1 (buttressing intact):** while the crevasse-to-thickness ratio
  `D(t)/H(t)` stays below the segment's feedback-onset ratio `phi`, the rift
  deepens at the segment's own base rate: `dD/dt = c0`.
- **Phase 2 (buttressing lost):** the instant `D(t)/H(t)` reaches `phi`, the
  neighbouring shelf can no longer brace the rift against ocean flow —
  back-stress is lost and growth accelerates: `dD/dt = c0 * (1 + BETA)` for
  the rest of the segment's life, where `BETA = 3.0` is a fixed structural
  gain shared by every segment on every shelf.

**Calving happens the instant `D(t)/H(t)` reaches a critical ratio `kappa`.**
`kappa` is a property of the particular shelf you are looking at — it is the
SAME for every segment in one test case, but it is **never given to you**;
you must infer it from the training table.

## Input (stdin)

```
n t
H0[0]  D0[0]  c0[0]  gamma[0]  phi[0]  T[0]
H0[1]  D0[1]  c0[1]  gamma[1]  phi[1]  T[1]
...
```

`t` is the test id; `n` training rows follow. Each row is one already-calved
segment: its initial thickness `H0`, initial crevasse depth `D0`, base
crevasse-growth rate `c0`, thinning rate `gamma`, feedback-onset ratio `phi`,
and the (noisily observed) time `T` at which it actually calved.

The training segments were logged with `phi` sitting just **below** `kappa`:
buttressing loss engages only in the closing moments before calving, if at
all, so most of each segment's life looks like plain phase-1 thinning and the
observed calving horizons are long.

## Output (stdout)

One line: a closed-form Python expression for `T` in the variables `H0`,
`D0`, `c0`, `gamma`, `phi`. Allowed: `+ - * / **`, unary `-`, numeric
constants, and the functions `sqrt log exp absv minv maxv`. Example
(illustrative **form only — NOT the hidden law**): `H0 / (c0 + gamma) - D0 *
phi`. No other names are accepted.

## Scoring (deterministic, minimization)

Your expression is evaluated on a **held-out table**, regenerated inside the
grader, whose segments have `phi` sitting a large, fixed margin **above their
own starting ratio `D0/H0`**: buttressing loss engages well before calving,
so a large share of each held-out segment's life runs at the accelerated
phase-2 rate — a regime the training table (where `phi` sits just below
`kappa`) essentially never visits. Let `p_i` be your prediction and `t_i` the
true (noisy) calving time at held-out row `i`:

```
metric   = mean_i  min(CAP, |ln(max(p_i, eps)) - ln(t_i)|)   # bounded log err
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant geomean(train T)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error gives a higher `Ratio` (capped at `1.0`). A constant
scores about `0.1`. `LAMBDA` is a small parsimony weight. Non-finite
predictions score `0`.

## Why the obvious fit is a trap

Fitting `kappa` from the training rows using only the phase-1 relation
`T = (kappa*H0 - D0) / (c0 + kappa*gamma)` — treating the rift as growing at
its OBSERVED rate forever — looks fine on training data, since those segments
barely touch phase 2. On the held-out table, where a large share of each
life is accelerated, this rate-only extrapolation predicts calving too late:
it never notices the trigger is the RATIO `D(t)/H(t)` crossing `phi`, not the
raw rate. Recovering `kappa` correctly means inverting the FULL two-phase
relation — using `phi` to locate the regime switch, not as one more linear
input to a curve fit — after which the fixed `BETA` predicts how much faster
the rift closes the remaining gap.

## Constraints

- Time limit 5 s, memory 512 MB; `n` is a few dozen rows.
- Held-out observation noise leaves irreducible error, so even the exact law
  does not reach `Ratio = 1.0` — there is room above the reference
  solutions.
- Scoring is fully deterministic.
