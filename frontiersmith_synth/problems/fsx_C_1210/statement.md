# The Lift That Fades After Two Weeks

A feature is rolled out to users in **cohorts** that enter treatment on
different calendar days (staggered adoption — cohort 1 first, cohort 2 six
days later, and so on). Every day, each currently-active cohort's treatment
**lift** (treated minus control, as a fraction) is measured.

A cohort's true lift is **highest right when it enters** — users are curious,
click around, explore the new thing (novelty / primacy) — and **decays**
smoothly toward a smaller **persistent** long-run lift as the cohort ages.
On top of that, every cohort visible **on the same calendar day** is also
nudged by a **common company-wide wobble** (traffic mix, a promo, a slow news
day) that depends only on the calendar day, not on how old any cohort is.

You see only the first two weeks (16 measured ages) of each of 6 staggered
cohorts. You must recover the **shape of the lift as a function of a cohort's
age**, because it will be graded far beyond that window — long after the
novelty has had time to fully fade.

## Input (stdin)
- Line 1: `n_rows test_id`.
- Next `n_rows` lines: `cohort entry_day calendar_day age lift`, one
  measurement each (`age = calendar_day - entry_day`, all integers except
  `lift`, a float). Rows are grouped by cohort; within a cohort, sorted by
  age.

## Output (stdout)
One line: a closed-form Python expression for the lift, in the single
variable `age`. Allowed: `+ - * / **`, unary `-`, numeric constants, and the
functions `sqrt log exp sig tanh absv`. Example (illustrative **form only —
NOT the hidden law**): `0.05 + 0.02 / (1.0 + age)`. No other names accepted.

## Scoring (deterministic, maximization)
Your expression is evaluated on a **held-out set of ages**, regenerated
inside the grader, that lie **far beyond the visible two-week window** — deep
in the regime where the persistent lift dominates. Let `p_i` be your
prediction and `L_i` the true (noisy) held-out lift at held-out point `i`:

```
metric   = mean_i  min(1, |p_i - L_i| / (|p_i| + |L_i|))     # bounded rel. error
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant predictor mean(train lift)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error gives higher `Ratio` (capped at `1.0`). The constant
baseline scores about `0.1`. `LAMBDA` is a small parsimony weight penalizing
an overgrown expression. Non-finite or complex-valued predictions score `0`.

## Why the obvious estimate is a trap
Averaging whatever you can see — even just the age-tail of the window, on
the reasoning that "recent measurements are closer to steady state" — still
carries an un-removed fraction of the novelty spike (the decay time constant
is often comparable to, or longer than, two weeks) and is still shifted by
the common calendar wobble that day. It **over-predicts** the long-run lift,
often badly.

Staggered entry is the way out. On any single calendar day, several cohorts
of *different ages* are visible **simultaneously**, and they all feel the
*exact same* calendar wobble that day. Difference the lifts of two cohorts
observed on the same calendar day and the wobble (and the persistent lift)
cancels exactly, leaving a signal that depends only on the two ages and the
decay curve. Fit the decay from these cross-cohort, same-day differences,
strip it back out of every row, and only then average what's left to
recover the persistent lift — that generalizes to the held-out horizon
where naive averaging does not.

## Constraints
- Time limit 5 s, memory 512 MB; `n_rows` is 96 (6 cohorts x 16 ages).
- Held-out noise and an un-forecastable future calendar wobble leave
  irreducible error, so even a correct decay/persistence split does not
  reach `Ratio = 1.0` — there is room above the reference solutions.
