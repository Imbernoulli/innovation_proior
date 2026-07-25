# Storm-Season Harbour Congestion

A harbour authority logs berth congestion for `K` vessel classes (tankers,
bulkers, ferries, ...). Each log row records, for one tide window:

- `rho` — total berth utilization (fraction of capacity demanded),
- `m_1 .. m_K` — the mix: class `c`'s share of the total demand,
- `w_1 .. w_K` — the observed mean queueing wait of each class.

You are given a **calm-season** log only: utilization stays low
(`0.08 <= rho <= 0.55`). You will be graded on a **storm-season** log you
never see — same harbour, same hidden dispatch mechanics, but utilization
`0.85 <= rho <= 0.97`, i.e. near capacity. Your predictors must extrapolate.

## Input (stdin)

```
n K t
rho m1 ... mK w1 ... wK        <- n rows
```

`t` is the test id; each test is a different harbour (different hidden
constants). All numbers are decimals; the mix fractions sum to 1.

## Output (stdout)

`K` lines, one per class:

```
W1 = <expr>
...
WK = <expr>
```

`<expr>` predicts that class's mean wait from the season variables.
Available variables: `rho`; `m1..mK`; `r1..rK` (class load `r_c = rho*m_c`);
`h` (mix concentration `m1^2 + ... + mK^2`). Allowed syntax: `+ - * /`,
parentheses, numeric constants, and the unary functions
`sig tanh relu absv exp log sqrt`. Limits: at most 120 nodes per expression
and 400 nodes total.

## Feasibility

Every line must match `Wc = <expr>` with each label `W1..WK` present exactly
once; every expression must parse under the grammar above (known names only,
finite numeric constants). While the grader evaluates your expressions on the
storm-season rows, every prediction must be finite and strictly positive.
Any violation scores `Ratio: 0.0`.

## Objective (minimise) and Scoring

On each storm-season grading row and each class,

```
e = min(1, |pred - w| / w)
```

Let `E` be the mean of `e` over all grading rows and classes, and `nodes`
your total node count. The grader forms

```
F = E * (1 + 0.002 * nodes)
B = E_baseline * (1 + 0.002 * K)
Ratio = min(1000, 100 * B / F) / 1000
```

where the internal baseline predicts each class's **calm-season mean wait**
(so a constant-per-class predictor reproduces `B`, Ratio ~ 0.1). The
storm-season rows are regenerated deterministically inside the grader from
the test id; scoring is bit-for-bit deterministic. Lower held-out error
raises the score; the small node tax discourages needlessly large formulas.

## Worked example (illustrative FORM only — NOT the hidden law)

```
W1 = 0.05 + 0.30 * rho + 0.80 * rho * rho
W2 = relu ( 0.02 + 0.10 * r2 + 0.40 * rho * rho ) + 0.001
```

This only shows the syntax. A smooth polynomial like this is exactly the kind
of calm-season curve fit that achieves excellent in-sample error — and the
harbour's true dispatch law has a very different shape. You must discover
that shape from the data.

## Hints from the dock

Veteran dispatchers insist the harbour does **not** serve vessels in arrival
order: some classes routinely overtake others at the berth, and a class's
wait depends on how much demand is queued *ahead* of it. They also warn that
waits "go vertical" as the harbour nears full capacity, and that storm
windows tend to draw a sparser, less balanced mix of classes than calm ones.
The calm-season log contains no high-load rows, yet the same mechanics govern
them — the calm rows quietly encode both the near-capacity behaviour and the
berth ordering, if you read them through the right functional form.

## Constraints

Time limit 5 s, memory 512 MB. `n <= 90` rows, `3 <= K <= 5`. Input files are
small. Scoring is fully deterministic.
