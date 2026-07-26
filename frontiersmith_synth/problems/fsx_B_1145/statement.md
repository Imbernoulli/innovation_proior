# One Guide, Three Kinds of Mountains

You are a rewrite guide leading a `term` — a list of integer symbols — down to the
shortest form you can reach, using two kinds of moves drawn from a ruleset given in
the instance:

- **Merge** `(a, b) -> c`: adjacent symbols `a` then `b` become the single symbol `c`.
  The term gets **one shorter**.
- **Split** `(a) -> (b, c)`: a single symbol `a` becomes the adjacent pair `b, c`. The
  term gets **one longer**.

You submit a sequence of moves (at most `budget` of them); each names its 0-indexed
**position** in the term *at the time it is applied* and which rule (by index) it
uses. A move is legal only if the symbol(s) there currently match the rule's
left-hand side exactly. Your score is the length of the final term after applying
your whole sequence in order — shorter is better.

The ruleset is drawn from a **mixture of three landscape shapes**, and you are not
told in advance which one you're facing — you have to read the structure yourself:

- **Gentle slopes.** Every merge rule is a plain same-symbol collapse (`a, a -> a`).
  Applying *any* legal merge, in *any* order, until none remain, always reaches the
  same shortest term (one symbol per maximal run of repeats). No cleverness needed.
- **Canyons.** Somewhere in the term sits a color-run, a lone "lock" symbol, then
  another run of the *same* color — the lock matches **no merge rule at all**, only
  its own split rule turning it into a same-colored pair. Refusing every
  size-increasing move leaves the lock stuck between two stub runs forever. Splitting
  it *first*, even though that one move lengthens the term, fuses both runs into one
  long same-color run that then collapses far below anything merge-only play reaches.
- **Braided trails.** Some adjacent pair matches **two different merge rules at once**
  (each move you submit names exactly one). Both shrink the term by the same one
  symbol this instant, so the immediate move can't tell you which is better — but one
  rule is a dead end (nothing further matches nearby afterward), while the other opens
  a "bridge" symbol that keeps absorbing what follows and can react with a distant
  trailing symbol for one more collapse later. Only downstream consequences tell them
  apart.

A rule of thumb tuned for one shape routinely wastes the other two.

## Public instance (stdin JSON)

```json
{
  "n": 24,
  "term": [2, 2, 0, 3, 3, 3, 1, ...],
  "budget": 58,
  "merges": [ {"a": 2, "b": 2, "c": 2, "family": 0}, ... ],
  "splits":  [ {"a": 7, "b": 3, "c": 3}, ... ]
}
```
`term` has exactly `n` integer symbols. `merges`/`splits` are the FULL, exact ruleset
(nothing hidden) — `family` is informational only (not read by the checker); it can
help you spot when two rules share a left-hand side. `merges`/`splits` list indices
are what your moves reference.

## Answer (stdout JSON)

```json
{"moves": [{"op": "merge", "pos": 4, "rule": 2}, {"op": "split", "pos": 1, "rule": 0}, ...]}
```
A list of at most `budget` moves, applied to the term in the given order.

## Feasibility

Each move must be a dict with `op` in `{"merge","split"}`, integer `pos`, and integer
`rule` indexing the corresponding ruleset. When a move is applied, `pos` must be in
range for the *current* term, `rule` must index an existing rule, and the symbol(s) at
`pos` must exactly match that rule's left-hand side. Any violation — out-of-range
index, mismatched symbol, wrong types, more than `budget` moves — scores `0` on that
instance. An empty move list is always feasible.

## Scoring

The evaluator computes baseline `b` = the length of the original term (doing nothing).
For a feasible move sequence producing a final term of length `obj`:
```
r = min(1, 0.1 * b / obj)
```
So doing nothing scores exactly `0.1`, and a final term `k` times shorter than the
original scores `min(1, 0.1*k)`. The reported `Ratio` is the mean of `r` over 10
deterministic, seeded instances (some larger and held out for generalization).
Infeasible or malformed answers score `0` on that instance.

Your program reads one public instance JSON from stdin and writes one answer JSON to
stdout. It runs in an **isolated subprocess** and only ever sees the public instance.

## Constraints

`20 <= n <= 65`. Time limit 5s.
