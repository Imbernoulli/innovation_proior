# The Ceiling You Can't See Yet — Forecasting Adoption Through a Competitor's Launch

## Problem
Product **A** is being adopted week by week. Cumulative adopters `A(t)` grow
under a fixed but hidden **diffusion-adoption** law: each step, a fraction of
the *remaining headroom* to some finite market ceiling converts into new
adopters. Early on, when `A(t)` is a small sliver of that ceiling, this
process is numerically almost indistinguishable from **pure exponential
growth** — the ceiling term is invisible in a short early window even though
it is always mathematically present.

At a step `t_B` that is **publicly known in advance** (launch dates are
filed with the exchange), a substitute product **B** goes on sale and starts
drawing from the *same* addressable market. From `t_B` onward, A's own
ceiling drops — some fraction of A's remaining headroom migrates to B
(**substitution-cannibalization**). Your job: recover a closed-form law for
`A(t)` from an early window that ends *before* `t_B`, accurate on a
**held-out horizon** that reaches past it.

## Input (stdin)
```
tid  n
tB  s_hint  M_hint
t_1  A_1
...
t_n  A_n
```
`tid` is the test id, `n` the number of training weeks (`t = 1..n`, in
order). `tB` is the exact, publicly known week B launches (`> n`). `s_hint`
is a noisy analyst estimate of the fraction of A's remaining headroom that
migrates to B. `M_hint` is a noisy analyst estimate of A's organic
(no-competition) ceiling. Neither hint is exact. Then `n` rows `(t, A_t)`,
noisy adoption counts.

## Output (stdout): a closed-form law
Emit one expression for `A` as a function of `t`. Allowed: numeric
constants, operators `+ - * /`, unary `+/-`, parentheses, the variable `t`,
the functions `expv(a)`, `logv(a)`, `sqrtv(a)`, `absv(a)`, `minv(a,b)`,
`maxv(a,b)`, `powv(a,b)` (`a` to the power `b`; `a` must evaluate positive),
and **at most one** top-level conditional
`EXPR1 if t <cmp> C else EXPR2` (`<cmp>` one of `< <= > >=`).

**Illustrative FORM only — NOT the hidden law:**
```
120.0 + 3.0*sqrtv(absv(t)) - 0.4*t if t < 20.0 else 900.0 - powv(t, 0.5)
```
This only shows the syntax; the real law's shape and constants must be
discovered from the data and the hints.

## Feasibility
The expression must parse under the grammar above (only known names/calls,
correct arities, finite constants, at most 200 expression nodes, at most one
comparison). Any parse violation, or any non-finite or non-positive value
produced while evaluating the law on the grading grid, scores `0`.

## Objective (maximise)
Let `pred_k` be your law at held-out step `t_k`, and `true_k` the (noisy)
true adopter count there. The grader forms the mean **squared LOG error**
(it rewards matching the trajectory's shape/rate, not just one level) plus a
small parsimony tax on expression size `nodes`:
```
F = mean_k (log(pred_k) - log(true_k))^2 * (1 + LAMBDA * nodes)
B = mean_k (log(A_n)    - log(true_k))^2 * (1 + LAMBDA * 1)   # A_n = your
                                            # LAST training value, frozen
Ratio = min(CAP, 0.1 * (B / F) ** GAMMA)
```
with small fixed constants `LAMBDA, GAMMA` (`0 < GAMMA < 1`) and a cap
`CAP < 1` so the score never saturates. Freezing the last training value
reproduces `B/F = 1` (Ratio = 0.1). Held-out observation noise, plus the
mismatch between the true step-by-step process and any smooth closed form,
keep even a strong law below the ceiling. Report the highest Ratio you can.

## Why the early window is a trap
Inside the training window the ceiling term contributes almost nothing — a
single log-linear fit (pure exponential) tracks the data beautifully. But
that fit has *no ceiling at all*: projected forward it keeps compounding,
oblivious to `t_B`, `s_hint`, `M_hint`. The true curve bends toward whatever
ceiling is in force, and that ceiling itself steps DOWN at `t_B`. Two
forecasters with the *same* early growth-rate estimate can land on wildly
different held-out scores if only one of them gets the *ceiling* right on
both sides of the launch — the rate says how fast you approach a level; the
level decides who wins the held-out window.

## Constraints
Time limit 5 s, memory 512 MB. `n = 13` training rows; held-out window has
11 steps spanning the launch. Scoring is fully deterministic.
