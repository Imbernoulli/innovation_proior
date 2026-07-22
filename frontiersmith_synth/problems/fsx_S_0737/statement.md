# The Flickering Lantern Row

A watchtower keeper strings `W` lanterns around a circular parapet (lantern
`W-1`'s right neighbour is lantern `0`). Every tick, ALL lanterns update
simultaneously: lantern `i`'s new state depends only on lanterns `i-1`, `i`,
`i+1` (its state before the tick) through one fixed, hidden **local rule**.
Nobody wrote the rule down.

You are given the keeper's logbook: a handful of **sparse, irregularly spaced**
readings (never two consecutive ticks apart) of the full row. Worse, the
keeper reads lanterns from the ground at night, and a small percent of
readings are misreported (lit read dark, or vice versa) every time he looks.
Your job: recover the rule well enough that it predicts the row **hundreds of
ticks later**, from a fresh starting row you never get to see mid-flight.

## Input (stdin)

```
t W m
tick_0 row_0
tick_1 row_1
...
tick_{m-1} row_{m-1}
```

`t` is the test id, `W` the row length, `m` the number of logbook entries.
Each `row_i` is a length-`W` string of `0`/`1`. Ticks are strictly increasing
and non-consecutive (`tick_{i+1} - tick_i >= 2`).

## Output (stdout): one boolean expression

Print a single line: a boolean expression over three variables `cL`, `cM`,
`cR` (the left neighbour, the cell itself, and the right neighbour — each
`0` or `1`). This expression **is** your candidate rule: it will be evaluated
on all 8 neighbourhood patterns to build an 8-entry rule table, then rolled
forward tick by tick.

Allowed: the operators `and or not ^ == != ( )`, the constants `0` and `1`,
and the three variable names above. Nothing else — no other names, no other
operators, no function calls.

**Illustrative FORM only — NOT the hidden rule:**

```
( cL == 1 and cR == 0 ) or not cM
```

This just shows the syntax. The real rule for each `t` is a different,
independently chosen radius-1 automaton rule; you must discover it from the
data.

## Feasibility

Your line must parse under the grammar above using only `cL`, `cM`, `cR`, and
must evaluate to exactly `0` or `1` on every one of the 8 neighbourhood
patterns. Any parse failure, disallowed syntax/name, or a value outside
`{0,1}` scores `0`.

## Objective (minimise)

The grader independently reconstructs, purely from `t`: the true rule, and a
**held-out** starting row (never shown to you) that it rolls forward `L=500`
ticks under the true rule, then applies the SAME misreading noise rate used
in the logbook to get an "observed" final row. It rolls your candidate rule
forward from the *same* held-out starting row for `500` ticks (no noise
added to your prediction) and compares:

```
mismatch = (fraction of the 500-tick-later row where your prediction
            disagrees with the observed final row)
F = mismatch * (1 + LAMBDA * nodes)          # nodes = expression size
B = mismatch_of_"the row never changes"       * (1 + LAMBDA)   # baseline
Ratio = min(1000, 100 * B / F) / 1000
```

Guessing "nothing changes" reproduces `B` (Ratio ≈ 0.1). Because the
comparison target itself carries observation noise, even the *exact* rule
cannot reach zero mismatch — it settles near the noise floor, leaving
headroom above any reference solution.

## Why the calm reading is a trap

Fitting "the cell 3-window at an earlier logbook entry predicts the cell at
the next entry" looks like sound statistics — more logbook pairs, more
training examples. But the true rule only maps a 3-window to the *very next*
tick. Across a gap of `k >= 2` ticks, a cell's true dependency has widened to
a `2k+1`-wide window; the same 3-window you can see will land on *different*
outcomes depending on cells you can't see, for many positions. A same-step
fit absorbs this as noise and drifts toward guessing. Recovering the rule
instead means treating it as **search over the 256 possible radius-1 rules**,
scoring each candidate by how well it explains *every* logbook gap when
rolled forward exactly that many ticks.

## Constraints

`W` up to ~190, `m = 5` logbook entries, time limit 5 s, memory 512 MB.
Scoring is fully deterministic.
