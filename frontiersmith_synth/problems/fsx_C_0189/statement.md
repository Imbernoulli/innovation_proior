# Tide-Pool Truce: Extragradient Schedule Design for a Predator-Prey Saddle

## Story

In a rocky intertidal **tide pool**, a guild of grazers (`x`) and a guild of predators
(`y`) settle into an uneasy truce. The grazers pick abundances to *minimize* their exposure
while the predators pick foraging effort to *maximize* their catch. Their joint payoff is a
smooth **convex-concave saddle function**

```
  Phi(x, y) = 1/2 x^T A x  +  x^T B y  -  1/2 y^T C y  +  a^T x  -  c^T y ,
```

with `A, C` symmetric positive semidefinite (self-limitation within a guild) and `B` the
antisymmetric-flavored cross-guild coupling (grazing pressure). The ecological equilibrium
(the "truce") is the saddle point `z* = (x*, y*)`, the unique zero of the monotone
**equilibrium-gradient operator**

```
  F(z) = [ dPhi/dx ; -dPhi/dy ] = M z + q ,
  M = [[ A ,  B ],
       [-B^T, C ]],     q = [a ; c],     z = (x, y).
```

You are the tide-pool manager. Each season you may apply exactly **K nudges** to the
community state, starting from `z0`. Nudge `k` uses the **extragradient** template below
with a step size `eta_k` and a look-ahead extrapolation `gamma_k` that **you choose**:

```
  g_k     = F(z_k)              = M z_k + q          # gradient at current state
  w_k     = z_k - gamma_k * g_k                      # extrapolated look-ahead state
  z_{k+1} = z_k - eta_k * F(w_k) = z_k - eta_k (M w_k + q)
```

After the K-th nudge, how far is the community from a true equilibrium? That is measured by
the **final equilibrium-gradient norm** `|| F(z_K) ||_2`. **Design the schedule
`(eta_0..eta_{K-1}, gamma_0..gamma_{K-1})` that makes it as small as possible.** The
template is fixed; only your schedule varies. There is no closed form for the best finite
schedule -- constant, accelerated, and multi-phase schedules all trade off differently, and
stiff pools with strong coupling cannot be fully quelled within the budget.

## Task

You write a standalone program. It reads ONE JSON object (the public instance) from stdin
and writes ONE JSON object (your schedule) to stdout.

### Public instance schema (stdin)

```json
{
  "M":  [[float, ...], ...],   // d x d operator matrix, d = 2n (n per guild), n in [8..17]
  "q":  [float, ...],          // length-d offset vector
  "z0": [float, ...],          // length-d starting community state
  "K":  4                       // iteration budget (number of nudges)
}
```

You are given the full operator, so you may **simulate the template internally** to search
for a good schedule. You cannot set the iterates directly -- you only pick the 2K scalars.

### Answer schema (stdout)

```json
{
  "eta":   [float, ...],   // length K step sizes
  "gamma": [float, ...]    // length K extrapolation coefficients
}
```

Every entry must be a finite real number with `|value| <= 1e6`. Any wrong shape, wrong
length, or non-finite entry is rejected and scores 0. A schedule that makes the trajectory
diverge (non-finite iterates) also scores 0.

## Objective (minimize)

For your schedule the evaluator deterministically re-runs the fixed template from `z0` and
computes the objective

```
  obj = || M z_K + q ||_2 .
```

Lower is better. The baseline is the **do-nothing** objective `b = || M z0 + q ||_2` (the
gradient norm before any nudge).

## Scoring

Per instance the normalized score is

```
  r = min(1.0, 0.1 * b / max(obj, 1e-12)) .
```

A do-nothing schedule (`obj = b`) scores `0.1`; halving the gradient norm relative to the
budget-scaled baseline moves you up, and the score is capped at `1.0`. The reported
**Ratio** is the mean over 10 seeded instances; later instances are stiffer (worse
conditioning, stronger coupling) and are effectively held out -- no single constant step
does well across all of them. Scoring is fully deterministic: the harness re-runs and
requires an identical Ratio and per-instance Vector.

## Isolation

Your program runs in an isolated subprocess and only ever sees the public instance above.
The evaluator scores your answer in its own process; you cannot read its state.
