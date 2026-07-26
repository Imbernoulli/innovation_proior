# Dielectric Breakdown Crossover -- reading a mechanism switch off the edges of a safe test window

A materials lab measures the **dielectric breakdown voltage** `V` of a thin
insulating film against two rig settings: film **thickness** `d`
(nanometres) and ambient **temperature** `T` (kelvin). To avoid destroying
samples, every measurement stays inside a **safe mid-range window**:
`d` in `[40, 90]` nm, `T` in `[280, 340]` K. Each test id is a different
film/dielectric-stack combination with its own fixed, undocumented law.

The film fails via **whichever of two competing breakdown channels gives
way first** -- a weakest-link system, like two fuses in parallel:

* An **avalanche-like** channel: its own critical voltage rises with `T`
  and scales close to linearly with `d`.
* A **tunneling-like** channel: its own critical voltage *falls* as `T`
  rises and scales *sub-linearly* with `d`.

Both channels are individually **dimensionally sensible power laws** in `d`
and `T` (a fixed reference scale, fixed exponents), so each channel's own
voltage is finite, positive, and monotonic in `d` and `T` on its own. The
**observed** voltage is a smoothed, **continuous** weakest-link combination
of the two -- never a hard jump -- so it stays finite and positive
everywhere, but bends toward whichever channel is locally weaker.

Inside the safe window both channels are comparable in size, so the data
alone barely hints that two mechanisms exist. Recover a closed-form law
`V(d,T)` that stays accurate **far outside** the window, including at its
four **extreme corners**, where one channel has decisively taken over and
the true curve has saturated onto that channel's own asymptote -- an
asymptote a single smooth global fit never bends towards. Near two of those
corners the winning channel depends on the specific film (not a fixed
threshold in `d` or `T` alone) -- get there by combining the edge hints in
your data with the stated physical constraints (continuity, per-channel
monotonicity, dimensional sensibility), not a rule tuned to one instance.

## Input (stdin)

```
t  N
d_0  T_0  V_0
d_1  T_1  V_1
...
```

`t` is the test id, `N=140` measurement rows follow. The held-out grading
corners are **not** given to you.

## Output (stdout): a closed-form law

Emit a single expression for `V` as a function of `d` and `T`. Allowed:
numeric constants, `+ - * /`, unary `+/-`, parentheses, the variables `d`
and `T`, and the functions `absv(a)`, `minv(a,b)`, `maxv(a,b)`,
`powv(a,b)` (`a` to the power `b`; `a` must evaluate positive),
`expv(a)` (`e^a`, any finite `a`), `logv(a)` (natural log; `a` must
evaluate positive). At most 260 expression nodes.

**Illustrative FORM only -- NOT the hidden law:**

```
12.0 + 0.4*absv(T - 300.0) / (1.0 + 0.02*d)
```

This only shows the syntax; the real law's channels, exponents and
coefficients are different and must be discovered from the data.

## Feasibility

The expression must parse under the grammar above. Any parse violation, or
any non-finite or non-positive value produced while evaluating the law on
the grading grid, scores `0` for that test.

## Objective (minimise)

Let `pred_k` be your law at held-out `(d_k,T_k)` and `true_k` the (noisy)
true voltage there. The grader forms the mean **squared LOG error** (it
rewards matching the right asymptote, not just the scale) plus a small
parsimony tax on expression size `nodes`:

```
F = mean_k (log(pred_k) - log(true_k))^2 * (1 + LAMBDA * nodes)
B = mean_k (log(Vbar)   - log(true_k))^2 * (1 + LAMBDA * 1)   # Vbar = flat
                                            # geometric mean of YOUR OWN
                                            # training V values
Ratio = min(0.90, 0.1 * (B / F) ** GAMMA)
```

with small fixed constants `LAMBDA, GAMMA` (`0 < GAMMA < 1`), capped below
1 so the score never saturates. Predicting the flat training average gives
`B/F = 1` (Ratio = 0.1); a law with the right saturated asymptotes drives
`F` down, raising the Ratio. Measurement noise on both the training rows
and the held-out grid keeps even a strong law below the ceiling.

## Why the safe window is a trap

Inside the window both channels are close enough in size that a single
global power-law regression explains the data almost perfectly -- nothing
in-sample screams that a second channel exists. But the two channels'
opposite thickness- and temperature-sensitivities leave a faint, opposite
CURVATURE signature at the two opposite edges of the window (one edge bends
toward the avalanche-like asymptote, the other toward the tunneling-like
one). That curvature -- easy to dismiss as noise -- is the only in-window
evidence a second channel exists. Take it seriously: the observed voltage
behaves like the lower envelope of two latent power-law surfaces, not one
surface with a slightly wrong exponent.

## Constraints

Time limit 5 s, memory 512 MB. `N = 140` rows; scoring is fully
deterministic.
