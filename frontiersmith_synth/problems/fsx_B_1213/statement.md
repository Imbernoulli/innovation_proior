# Efficacy at a Dose No One Has Tried

A compound has been tested only at low doses. Your job: recommend the dose,
anywhere up to the widest ethically approvable dose `Dmax`, that gives the
best net clinical value -- including doses far beyond anything measured.

## The pharmacology (told honestly; the constants are hidden)

**Efficacy** comes from receptors binding the compound. Receptors are finite,
so efficacy **saturates**: it follows a Hill-type curve

```
E(d) = Emax * d^n / (EC50^n + d^n)
```

for unknown positive constants `Emax` (the efficacy ceiling), `EC50` (the dose
at half-maximal effect -- the **saturation constant**), and Hill coefficient
`n`. As `d -> infinity`, `E(d) -> Emax`: no matter how high you dose, efficacy
cannot climb past the ceiling.

**Toxicity** has no such ceiling in the range that matters here: it is
monotonically increasing and **accelerates** (its curve is not told to you --
only that it never levels off).

Because efficacy saturates and toxicity keeps climbing, the **net clinical
value** `U(d) = E(d) - T(d)` rises with dose at first, then eventually *falls*
-- the therapeutic window closes at high dose. All doses you get to observe
are low, well before the window closes; you must extrapolate to find where it
does.

## Input (stdin)

```
n_train  t  Dmax
d[0]  eff[0]  tox[0]
d[1]  eff[1]  tox[1]
...
```

`t` is the test id. Each of the `n_train` rows is a noisy measurement (dose,
observed efficacy, observed toxicity) taken at a **low** dose. `Dmax` is the
widest dose that will ever be considered -- most of it is territory you have
no direct measurements in.

## Output (stdout): two curve expressions

```
EFFICACY <expr>
TOXICITY <expr>
```

Each `<expr>` is an arithmetic expression in the single variable `d`, using
`+ - * / **`, parentheses, numeric constants, and the unary functions `exp`,
`log`, `sqrt`, `abs`. No other names or functions are allowed.

**Illustrative FORM only -- NOT the hidden law** (toy numbers, unrelated
shape):

```
EFFICACY 10 + 2*sqrt(d)
TOXICITY 0.5*d
```

This just shows valid syntax.

## How you are graded

The grader evaluates your two expressions at 241 doses spaced evenly across
the **entire** allowed range `[0, Dmax]` (almost none of which you were shown)
and finds `d_hat`, the dose where *your own* `EFFICACY - TOXICITY` is
largest. Any non-finite or invalid value anywhere on that grid scores 0.

It then checks how good `d_hat` **actually** is, using the real (hidden)
efficacy and toxicity curves:

```
frac  = ( Utrue(d_hat) - Utrue(0) ) / ( Utrue(d*) - Utrue(0) )     clipped to [0,1]
Ratio = 0.1 + 0.75 * frac
```

where `d*` is the true best dose on the same grid. Recommending "give
nothing" (`d=0`) always reproduces the floor `Ratio = 0.1`. Correctly locating
`d*` pushes the score toward the ceiling -- but measurement noise keeps even a
very good fit from hitting it exactly, so there is always headroom above a
good submission.

## Feasibility

Both expressions must parse under the grammar above and evaluate to a finite
real number at **every** grid dose. Any parse failure, disallowed name,
domain error (e.g. `log` of a non-positive number), or non-finite/complex
result anywhere on the grid scores `Ratio: 0.0`.

## Why the low-dose data is a trap

At the doses you can see, both curves still look like they are simply
"rising" -- there is no visible ceiling yet, and no obvious sign of how fast
toxicity will accelerate later. A curve fit with no ceiling built in (e.g. a
plain polynomial) will often keep predicting efficacy climbing all the way to
`Dmax`, recommending doses deep inside territory where toxicity has already
overwhelmed a long-saturated efficacy.

## Constraints

`n_train` is 17-26 rows. Time limit 5 s, memory 512 MB. Scoring is fully
deterministic.
