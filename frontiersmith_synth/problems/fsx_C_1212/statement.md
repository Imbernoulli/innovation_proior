# Free Flow Until One Brake Light

## Problem

A freeway's flow-density relation is not a single curve. Below a **critical
density** it is a classic increasing free-flow branch: more density means
(up to a point) more throughput. Near and above that critical density the
relation turns **metastable and history-dependent**: at the *same* density,
the road can sustain a high, near-capacity flow if undisturbed, or fall
onto a substantially *lower* post-breakdown flow if a disturbance — one
brake light rippling backward through the platoon — tips it into a moving
jam. This is the documented **capacity-drop / hysteresis** phenomenon: the
discharge flow out of a jam is measurably lower than the flow sustained
right before the jam formed, at that very same density.

You are handed a log recorded only while density stayed **comfortably
below** the critical point — genuinely free flow, single-valued. Each
logged moment also records a perturbation magnitude (`P`, a disturbance /
brake-event size) and the flow's response to it. At low density this
response is negligible; but the *size* of the reaction to a given `P`
grows the closer density creeps toward the critical point — a rising
sensitivity to disturbance that is the only hint, from safely sub-critical
data, of where and how sharply the road will misbehave once density moves
higher. You are graded on densities **above** anything you observed,
spanning the metastable/broken-down region, where a perturbation of a
given size present at that density can decide which branch is realized.

## Input (stdin)

```
n t
rho[0]  P[0]  q[0]
rho[1]  P[1]  q[1]
...
rho[n-1] P[n-1] q[n-1]
```

`t` is the test id. `n` training rows follow: density `rho` (sub-critical,
positive), perturbation magnitude `P` (float in `[0, 25]`), and observed
flow `q` (positive, with measurement noise). Held-out grading densities lie
in a **higher**, non-overlapping range for the *same* hidden road; not
given to you.

## Output (stdout): one closed-form expression

Print **one line**: an arithmetic expression over `+ - * / **`,
parentheses, numeric constants, and the variables `rho` and `P` only (no
function calls, no other names). It is evaluated directly as your
predicted flow at each held-out `(rho, P)`.

**Illustrative FORM only — NOT the hidden law:**
```
1200 + 3.5 * rho - 0.2 * rho * P
```
This just shows the syntax; the real relationship has a different shape
you must discover from the data.

## Feasibility

The expression must parse under the grammar above (only names `rho`, `P`;
finite constants; at most 150 nodes). Evaluated at every held-out point it
must give a finite, non-negative result — flow cannot be negative. Any
violation scores `0`.

## Objective (maximise)

Let `MSE` be the mean squared error of your expression's predictions
against the held-out flow, and `nodes` the number of expression nodes. The
grader forms

```
F = MSE * (1 + 0.01*nodes)
B = MSE_of_constant_mean_training_flow * (1 + 0.01*1)   # internal baseline: predict the mean training flow, ignoring rho and P
r = B / F
Ratio = 0.88 * r / (r + 7.8)
```

A submission that reproduces the baseline scores `Ratio ~= 0.10`. Lower
held-out error raises the score, but the map is a bounded curve: `Ratio`
never reaches `0.88` — held-out noise, plus the fact that the metastable
decline rate and the capacity-drop fraction both vary substantially per
road (never recoverable from sub-critical-only data), keep a perfect fit
out of reach. A needlessly large expression is taxed via `nodes`.

## Why the sub-critical data is a trap

Fit *any* single curve `q(rho, P)` to the sub-critical rows — even one
that gets the perturbation response about right at the densities you
saw — and it extrapolates as one smooth continuation. But past the
critical density the true relation **splits**: some held-out points sit on
a persisting near-capacity branch, others on a markedly lower discharge
branch *at the very same density*, and the split pulls flow far below any
naive continuation as density climbs deeper into the broken-down region. A
model that only ever fit a single global trend cannot represent that split
at all. Only the *growth rate* of the perturbation response as density
approaches the critical point — not its typical size in your visible
data — locates where and how hard that split bites.

## Constraints

`n` is 50–70 rows. Time limit 5 s, memory 512 MB. Scoring is fully
deterministic.
