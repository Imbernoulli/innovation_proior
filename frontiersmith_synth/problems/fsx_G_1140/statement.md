# Tropical Toll Roads — recovering a min-plus automaton that extrapolates

A toll network reads a string over the alphabet `{a, b}` symbol by symbol and
charges a **total toll** for the trip. The network has an unknown small number
of internal "lanes" (states); reading each symbol either keeps you in your
current lane at some per-symbol toll, or (for at least one symbol, in at least
one lane) lets you switch to a different lane, paying a one-time toll for the
switch itself. The total toll of a string is the **cheapest** possible sequence
of lane choices that reads the whole string start to finish — a *min-plus
(tropical) weighted automaton*: cost = min over accepting paths of the sum of
the weights along the path.

You are given a notebook of `(string, toll)` pairs recorded on **short** trips.
Your job is to output your own small min-plus automaton whose tolls
reconstruct the notebook and — more importantly — correctly predict tolls on
**much longer** trips that you never observed.

**Illustrative FORM only — not the hidden law.** A single-lane network that
always charges 3 per `a` and 5 per `b` costs `"abba"` exactly `3+5+5+3=16`,
regardless of order. The real hidden network need not look like this at all,
and may have more than one lane and switching behaviour.

## Input (stdin)
- Line 1: two integers `t n` (an instance id and the number of training rows).
- Line 2: `alphabet a b` (fixed, always these two symbols).
- Next `n` lines: `<string> <toll>`, one training observation each (`toll` is
  a non-negative integer).

## Output (stdout)
Your automaton, as whitespace-separated tokens (any layout / line breaks):
```
S T
i c j w        (T lines; state i, symbol c in {a,b}, target state j, weight w)
start
K
j f            (K lines; accepting state j, final weight f)
```
`1<=S<=8` states, `0<=T<=64` transitions, `|w|,|f|<=1e5`. Several transitions
may share the same `(i, c)` — the automaton is nondeterministic, and its cost
for a symbol sequence is the **minimum** total weight over all ways to read it
from `start` to some accepting state (standard min-plus / tropical semantics).
All weights are real numbers (not necessarily integers).

## Feasibility
Malformed tokens, out-of-range indices/sizes, non-finite weights, or zero
accepting states all score `0`. Your automaton must also **fit the training
data**: if your automaton's own predicted tolls deviate too much (mean
relative error) from the training tolls you were given, you score `0` — you
must genuinely model the notebook, not ignore it and hope for the best on the
long trips.

## Scoring (deterministic, maximize)
The grader regenerates, from the instance id alone, a **held-out set of much
longer strings** (well beyond any training length, plus a few pure-`b` long
strings), evaluates your automaton's tolls on them, and compares against the
true (slightly noisy) tolls:
```
O = mean_i min(5, |predicted_i - true_i| / max(1, true_i))      # your held-out error
B = same metric for a length-proportional flat-rate baseline
    fit from your training rows (ignores which symbol, ignores routing)
Ratio = clip(0.9 - 0.8 * (O / B), 0, 1)
```
Matching the flat-rate baseline scores about `0.1`; held-out sensor noise
means even an exactly-recovered automaton does not reach the `Ratio = 1.0`
ceiling — there is room above the reference solution.

## Why the obvious fit is a trap
Fitting one additive cost per symbol (linear regression on character counts)
reproduces the notebook fairly well, because on short trips one lane choice
usually dominates outright. But whether it is *ever* worth switching lanes
depends on **where** the option first appears and **how much trip remains
after it** — not on symbol counts alone. On far-longer held-out trips the
cheapest route switches lanes almost everywhere, and an additive model, blind
to this position-dependent regime change, misses it substantially. Only an
automaton that genuinely represents the lane-switch (a changepoint in how
cost grows with length, discoverable in the training data) tracks the true
cost out to unseen lengths.

## Constraints
- Time limit 5 s, memory 512 MB. Training rows: at most ~70. Held-out strings:
  at most a few hundred characters each.
