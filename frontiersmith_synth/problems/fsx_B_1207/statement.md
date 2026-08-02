# Resistance Sweep Forecast

A hospital ward tracks the fraction `p(t)` of a bacterial population that
carries a plasmid-borne antibiotic-resistance gene. Two forces continuously
push `p` around:

- **Introduction.** Mutation (rate `mu`) and horizontal plasmid transfer
  between cells (rate `tau`) constantly reintroduce resistant cells at a
  combined rate `(mu+tau)` per susceptible cell, regardless of any drug.
- **Selection.** Resistant cells pay a **fitness cost** `c` (slower growth,
  metabolic burden of carrying the plasmid) when no drug is present, but gain
  a survival advantage `alpha*D` under a drug applied at concentration `D`.

The frequency evolves as
`dp/dt = (mu+tau)*(1-p) + p*(1-p)*(alpha*D(t) - c)`.
`D(t) = 0` throughout the window you observe (surveillance is running
*before* any treatment decision). Then, at time `T0`, a course begins:
`D(t) = D` (a fixed dose) for the rest of the forecast horizon, up to
`T0+T1`. Everything above — `mu`, `tau`, `alpha`, `D`, `T0`, `T1` — is given
to you exactly. The fitness cost `c` is **not** given anywhere; the only clue
to it is the pre-treatment data.

*Illustrative FORM only, not this instance's numbers:* if a quantity `q(t)`
relaxed toward equilibrium `E` at rate `k`, you might write
`q(t) = E - (E-2.0)*exp(-k*t)` — the ACTUAL curve you need has a different
algebraic shape (see Scoring); you must derive it, not pattern-match this.

## Input (stdin)
```
testId
mu tau alpha D T0 T1
n_train
t_1 p_1
...
t_n_train p_n_train
n_query
q_1
...
q_n_query
```
The `n_train` rows are noisy resistant-fraction readings taken at times
`t_i < T0` (pre-treatment only). The `q_j` are the times (all `> T0`) at
which your forecast will be scored; their true values are held out.

## Output (stdout)
Print **one line**: a single closed-form arithmetic expression over the
variable `t`, using `+ - * / **`, parentheses, numeric literals, and the
unary functions `exp`, `log`, `sqrt`. No other names are allowed. The grader
evaluates your expression at each `q_j` (substituting `t = q_j`) to get your
forecast `p_hat(q_j)`.

## Feasibility
The expression must parse under the grammar above (only `t`, the three
listed functions, and numeric literals) and must evaluate to a finite number
at every query time. Any parse failure or non-finite value scores `Ratio:
0.0`.

## Objective and Scoring
Let `MAE` be the mean absolute error of your `p_hat(q_j)` against the true
held-out `p(q_j)`. The grader also computes `MAE_base`, the error of the
context-free constant forecast `0.5`. Then
```
F = 1 / (MAE + 0.1)
B = 1 / (MAE_base + 0.1)
Ratio = min(1000, 100*F/B) / 1000
```
Guessing `0.5` blind reproduces `B` (`Ratio ~ 0.1`). Lower held-out error
raises the score; the score saturates below 1.0, so there is always room for
a tighter fit.

## Why the flat pre-treatment line is a trap
Before treatment, `p` sits at a quiet equilibrium and looks like noise around
a constant — nothing in the visible window hints that anything will change.
Extrapolating that constant is the natural first move, and it is exactly
right **until** `D` turns on. Once it does, the same equilibrium level that
looked like "nothing happening" is actually pinned there by a specific
fitness cost `c`, and that `c` — not the flat value itself — is what
determines how fast (or whether) resistance sweeps once dosing starts.

## Constraints
`T0=20, T1=24`. `n_train` is 10–20, `mu,tau` are O(1e-3), `alpha` is O(1),
`D` and `c` keep `p(t) in [0,1]` for all queried times. Time limit 5 s,
memory 512 MB. All scoring is deterministic.
