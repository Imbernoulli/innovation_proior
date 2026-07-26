# The Glassblower's Hysteresis Loop

An old glassblower swears the final color of a piece depends on the whole
cooling *schedule*, not just the temperature it ends at — and the furnace
logbook backs her up. Your job: recover her hidden rule.

A hidden first-order transition governs the color measurement `m`. There
are two stable **branches**, `m_hot(T)` and `m_cold(T)`, both affine in the
temperature `T`. A piece always enters a schedule freshly pulled from the
furnace (molten, on the hot branch). While a schedule **cools**, the hot
branch loses stability and the piece **jumps** to the cold branch the
instant `T` drops through a lower spinodal temperature `T_down`. While a
schedule later **reheats**, the cold branch loses stability and the piece
jumps back to the hot branch the instant `T` rises through an *upper*
spinodal `T_up > T_down`. Between the two spinodals lies a **bistable
window**: for any `T` in `(T_down, T_up)`, the color depends on *which
branch you approached from* — the outcome is a property of the whole
temperature-vs-progress path `T(t)`, not of the endpoint alone.

Every training schedule you are given is **monotone cooling** (pulled hot,
cooled to some final temperature, never reheated). Under monotone cooling
the branch can only switch once, so on this data the final color happens to
be fully explained by the *endpoint* temperature — path and endpoint are
confounded. You will be graded on **held-out schedules that reheat**,
visiting the bistable window from both directions, where that confound
breaks.

**Illustrative FORM only — NOT the hidden law:**
`m_hot(T) = 10 + 0.001*T`, `m_cold(T) = 3 - 0.0005*T`, `T_down=500, T_up=520`.
This only shows the shape of the two-branch, two-threshold structure; the
real coefficients and spinodals are different and must be discovered from
data.

## Input (stdin)

```
T_MIN T_MAX N t
K  T_0 T_1 ... T_{K-1}  m_noisy      (N of these rows)
```
`T_MIN, T_MAX` bound the temperature domain; `t` is the test id; `N`
training rows follow. Each row is a schedule with `K` breakpoint
temperatures (strictly decreasing — `T_0` is always well above any possible
`T_up`, so the piece starts molten) connected by straight lines, followed by
a noisy measurement `m_noisy` of the final color reached at `T_{K-1}`.

## Output (stdout): a branch diagram

Print exactly three lines:
```
A1 B1        (your model of m_hot(T)  = A1 + B1*(T-600))
A2 B2        (your model of m_cold(T) = A2 + B2*(T-600))
Tdown Tup    (your estimated spinodal temperatures)
```

## Feasibility

All six numbers must be finite; `|A1|,|A2| <= 1000`, `|B1|,|B2| <= 5`;
`Tdown <= Tup`; both thresholds within `[-500, 1500]`. Any violation scores
`0`.

## Objective (minimise)

The grader regenerates a set of **held-out schedules** (never shown to you),
including reheats into and through the bistable window. For each one it
rolls YOUR model forward with the exact same branch-and-jump rule described
above (branch starts `hot`; switches to `cold` the instant a *decreasing*
segment crosses your `Tdown`; switches back to `hot` the instant an
*increasing* segment crosses your `Tup`) to get a predicted final color, and
compares it against the true one:

```
F = RMSE(your predictions, truth)
B = RMSE(constant = mean of your training m_noisy values, truth)
Ratio = min(1000, 100*B/F) / 1000
```

Ignoring temperature reproduces the baseline (`Ratio ~= 0.1`). A model that
fits both branch curves well but sets `Tdown == Tup` (no hysteresis loop)
does well on unambiguous schedules but is wrong throughout the bistable
window whenever a reheat doesn't fully cross back. A model with a genuine,
correctly-sized loop does much better — but `T_up` is never directly
exercised by monotone training data, so getting it exactly right, and
handling every reheat depth, stays out of reach.

## Worked example (tiny, illustrative)

Training: `(T_end=800, m=61)`, `(T_end=400, m=34)`. A model with
`A1=60,B1=0,A2=33,B2=0,Tdown=600,Tup=600` predicts `61` and `33` — small
training error but zero hysteresis. If the true `Tup=650`, a held-out
schedule `900 -> 350 -> 620` (dips cold, reheats to 620 without reaching
650) truly stays **cold** (`~33`); the zero-width model wrongly predicts
**hot** (`~61.3`).

## Constraints

Time limit 5 s, memory 512 MB. `N` up to a few hundred rows, each `.in`
well under 1 MB. Scoring is fully deterministic.
