# Collision-Rig Consortium — What Stays Constant When the Rigs Differ

A consortium operates **5 two-body collision rigs**. Rig `r` has fixed masses
`m1, m2`, a damping coefficient `g`, and a sampling interval `dt`. Each logged
row is one collision: pre-impact velocities `(v1, v2)` and the measured
post-impact velocities `(v1p, v2p)`. Post-impact readings carry sensor noise
whose scale grows with the rig's sampling interval.

Every rig obeys the **same hidden collision law**; only the rig constants
(masses, damping, sampling) differ. Your job: learn the law from the training
logs and predict post-impact velocities on a **new rig you never see** — one
whose masses lie **outside every training rig's mass range** (and whose damping
may also lie outside the training band). A curve that merely interpolates the
training clouds will not survive the new regime; a quantity that stays
**constant across every rig's collisions** will.

## Input (stdin)

```
<n_rows> <test_id> 5
RIG <m1> <m2> <g> <dt>            (5 lines, rig indices 0..4 in order)
ROW <rig> <v1> <v2> <v1p> <v2p>   (n_rows lines)
```

## Output (stdout): a predictor program

```
LET <name> <expr>     (optional, at most 8, evaluated in order)
V1 <expr>             (required: predicted post-impact velocity of body 1)
V2 <expr>             (required: predicted post-impact velocity of body 2)
```

`<expr>` is an arithmetic expression over:

- rig constants `m1`, `m2`, `g` (bound to the grading rig's values),
- pre-impact velocities `v1`, `v2` of the collision being scored,
- any previously defined LET names (a LET name is `[a-z][a-z0-9]{0,7}`, not a
  reserved word, defined at most once),
- numeric constants (finite, |c| <= 1e9),
- operators `+ - * / **` and unary `-`,
- unary functions `exp, sqrt, absv, tanh, sig`
  (`sig(x) = 1/(1+exp(-x))`; `exp`/`sig` saturate at |x|=60).

Total program size <= 200 syntax nodes. Both `V1` and `V2` must appear exactly
once. Every prediction must be finite for every graded collision.

## Feasibility

Any of the following scores 0: unparseable program, unknown names or
functions, disallowed syntax, missing/duplicated `V1`/`V2`, too many LETs,
node limit exceeded, non-finite constants, or a non-finite/erroring
evaluation on any graded collision.

## Scoring (deterministic)

The checker regenerates — from `test_id` alone — a held-out grading rig
(masses outside the training band) with 240 fresh collisions from the hidden
law plus seeded sensor noise, then:

```
F = MSE(your predictions)  * (1 + 0.001 * nodes)
B = MSE(identity baseline) * (1 + 0.001 * 2)     # predicts post = pre
Ratio = min(1000, 100 * B / F) / 1000            # higher is better
```

Predicting "nothing happens" scores ~0.1. Halving the baseline error doubles
the score; a perfect-law recovery is bounded above only by sensor noise and
any structure you failed to model. The parsimony coefficient is small but
real: shorter equivalent programs score higher.

## Example (illustrative FORM only — NOT the hidden law)

A "classical momentum + constant restitution 0.5" guess could be written:

```
LET mu m1 * v1 + m2 * v2
V1 ( mu + m2 * 0.5 * ( v2 - v1 ) ) / ( m1 + m2 )
V2 ( mu - m1 * 0.5 * ( v2 - v1 ) ) / ( m1 + m2 )
```

This particular shape (unit exponent, one constant restitution for all rigs)
is shown only to demonstrate the program syntax. The true law's exponent,
damping dependence, and any further structure must be **discovered from the
data** — e.g. by testing candidate conserved quantities for constancy on
every rig before fitting anything.

## Constraints

- 5 rigs; ~75–300 rows per rig (faster sampling → more rows, less noise).
- Training masses in [1, 8]; the grading rig's masses are strictly above 9.
- Time limit 5 s, memory 512 MB. Deterministic scoring: same program, same
  score, always.
