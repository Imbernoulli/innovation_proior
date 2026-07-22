# The Quiet Chain: Weighted Constraints with a Hidden Backbone

You are given `n` boolean variables, numbered `1..n`, and a list of weighted OR
**clauses**. Each clause is a short list of signed integer literals: a positive literal
`v` is satisfied when variable `v` is `True`; a negative literal `-v` is satisfied when
variable `v` is `False`. A clause is satisfied if **any** of its literals is satisfied.
You must output **one full assignment** of all `n` variables.

## Input (one JSON object on stdin — the public instance)

```
{"name": str, "n": int,
 "clauses": [[lit, lit, ...], ...],
 "weights": [w, w, ...]}
```

`weights` has the same length as `clauses`; `weights[i] > 0` is the value of satisfying
`clauses[i]`. Weights vary a lot across the list — a handful of clauses carry weight far
above the rest.

## Output (one JSON object on stdout)

```
{"assign": [a_1, a_2, ..., a_n]}
```

Each `a_i` must be literally `0` or `1` (not `true`/`false`, not a float). Wrong length,
any non-`{0,1}` entry, a crash, a timeout, or non-JSON output scores `0` on that instance.

## Objective (maximize)

```
obj = sum of weights[i] over all clauses[i] that are satisfied by your assignment
```

## What to notice

Weight is not spread evenly. A small set of clauses — some with just one literal,
some with two — carry weight so far above everything else that satisfying them is
essentially non-negotiable: whatever assignment you pick for the variables they touch,
getting these few clauses right is worth more than getting a great many of the small
ones right. The two-literal heavy clauses chain together: satisfying one forces what the
next one needs, and so on — you can work out the *entire* required value for every
variable those heavy clauses reach with nothing more than propagation, before you touch
any kind of search. You never have to guess at these variables.

Meanwhile, a much larger population of low-weight clauses connects many other
variables in the ordinary way — no single one matters much, but collectively they are
what genuinely rewards careful local search (there is no shortcut that reads them off
directly). Some of these small clauses also happen to touch the same variables the heavy
clauses pin down — and are individually cheaper to satisfy if that variable sits at the
*other* value. Chasing those small, plentiful rewards one flip at a time, without ever
stepping back to notice which variables are pinned and why, is a good way to quietly give
up a lot of weight you didn't have to.

## Scoring (deterministic; no wall-time)

For each of 10 fixed seeded instances, let `b` be the score of a static per-variable
majority heuristic (computed by the evaluator itself, independent of your program — for
each variable, whichever polarity has more total clause-weight backing it) and `hi` a
generous, **unreachable** upper bound (the sum of every clause's weight, times a small
slack factor — the clause population is dense enough that no assignment satisfies them
all simultaneously):

```
r = clamp( 0.1 + 0.9 * (obj - b) / (hi - b), 0, 1 )
```

Your score is the **mean of `r`** over the 10 instances: calmer instances with only a
mild pull away from the heavy clauses, sharper "trap" instances where that pull is
strong, and held-out instances with a wider or deeper hidden chain for generalization.

## Constraints

- `1 <= n <= 200`, each clause has 1–3 literals, `1 <= weights[i] <= 40`.
- Time limit: a few seconds per instance. Read stdin once, write one JSON object to
  stdout, and exit.
