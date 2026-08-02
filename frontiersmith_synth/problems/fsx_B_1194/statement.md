# Wake Onset Forecast

A cylinder sits in a cross-flow controlled by one parameter `R` (a Reynolds-like
flow number). Below some critical value `Rc` the wake is **steady**: any small
perturbation you inject dies out. Above `Rc` the wake **spontaneously sheds
vortices** — it locks onto a self-sustained oscillation. This qualitative
switch is a supercritical Hopf bifurcation, and near `Rc` it has a universal
local shape.

The least-stable mode has a growth rate that is (locally) linear in the
control parameter:

```
g(R) = a * (R - Rc)          a > 0, both a and Rc unknown to you
```

- **Below onset** (`g(R) < 0`, i.e. `R < Rc`): the flow is steady. A small
  perturbation injected into it decays; its measured **decay rate** is
  `d(R) = -g(R) > 0`. The oscillation amplitude itself is zero (plus sensor
  noise).
- **Above onset** (`g(R) > 0`): the mode grows until nonlinearity saturates it
  on a limit cycle. The equilibrium oscillation **amplitude** obeys the
  standard Landau law
  ```
  A(R) = sqrt( g(R) / L )
  ```
  where `L > 0` is a fixed, facility-specific constant (given to you) that
  sets how fast the cubic nonlinearity saturates the growth. For `R <= Rc`,
  `A(R) = 0`.

## Input (stdin)

```
M t
L
R[0]  decay_rate[0]  amplitude[0]
...
R[M-1] decay_rate[M-1] amplitude[M-1]
```

`t` is the test id, `L` the Landau constant. Every logged `R[i]` in this
notebook was measured **below** the (unknown) `Rc`: `decay_rate[i]` is the
measured decay rate of an injected perturbation there, and `amplitude[i]` is
the measured self-sustained oscillation amplitude — always noise around zero,
because the flow never left the steady branch during this campaign. You are
graded on a flow-parameter range you never got to test, including values well
past onset.

## Output (stdout)

One line: a closed-form Python expression for the amplitude `A` as a function
of `R`. Allowed: `+ - * / **`, unary `-`, numeric constants, and the functions
`sqrt log exp sig tanh relu absv`. Example (illustrative **form only — NOT the
hidden law**): `0.4 * tanh(0.02 * (R - 90))`. No other names are accepted.

## Scoring (deterministic, minimization)

Your expression is evaluated on a held-out grid of `R` values, regenerated
inside the grader, spanning subcritical, just-past-onset, moderately, and
**far** past-onset flow parameters. Let `p_i` be your prediction and `t_i` the
true (noisy) amplitude at held-out point `i`:

```
MSE      = mean_i (p_i - t_i)^2
O        = MSE * (1 + LAMBDA * nodes)          # nodes = expression size
baseline = MSE of the constant predictor 0 (i.e. "the wake never sheds")
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error raises `Ratio` (capped at `1.0`). A constant-zero
predictor scores about `0.1`. `LAMBDA` is a small parsimony weight. Non-finite
predictions score `0`.

## Why the amplitude column is a trap

Fitting the `amplitude` column directly — with anything, linear, sigmoid,
whatever — reproduces the constant zero, because *every* training row lies on
the steady branch: there is no signal about the post-onset shape in it at all.
That fit scores exactly the baseline and can never beat it.

## Why extrapolating the decay rate alone still isn't enough

The `decay_rate` column *does* carry a signature of the approach to onset: it
falls linearly toward zero as `R` climbs toward `Rc`, and that same line
(flipped in sign) is the growth-rate law `g(R)` that keeps governing the flow
past onset. But `g(R)` is a **growth rate**, not an amplitude — reporting it
as-is ignores the cubic saturation that `L` encodes, so it badly under-sizes
the amplitude just past onset (where `sqrt` rises steeply but `g` is still
tiny) and drifts further out. You need the `sqrt(g(R)/L)` relation, applied to
a growth-rate line honestly extrapolated from where it's actually visible:
the subcritical decay data.

## Constraints

Time limit 5 s, memory 512 MB; `M` is at most a couple dozen rows. Held-out sensor noise
leaves irreducible error, so even a correct law does not reach `Ratio = 1.0`.
