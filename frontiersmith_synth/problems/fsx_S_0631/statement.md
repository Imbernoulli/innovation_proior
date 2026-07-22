# Bonsai Ledger: Pruning Between Visitor Seasons

## Story

A bonsai garden displays its specimens along the search path of a binary
search tree keyed by tag number. Visitors ask the gardener to find tag `k`;
the gardener walks pointers from the display root down to it, one hop at a
time. Between look-ups the gardener may **prune-and-reset a branch** — a
single BST rotation that reshapes the display — to bring a frequently
requested specimen closer to the entrance. But re-staging a branch takes real
labor, and during a busy visitor **season** every rotation costs far more
(staff, foot traffic, safety rope) than during a quiet one. The trace tells
you exactly how costly each moment's pruning is — read and exploit it.

You write a policy that, for the whole day's trace, decides **before each
visitor's search** which tags to rotate up, and by how much.

## Input (stdin, one JSON object — the public instance)

```json
{"name": "spring_procession_peak", "n": 400,
 "left": [ ... n ints ... ],   "right": [ ... n ints ... ],
 "root": 173,
 "season_weight": [ ... M numbers ... ],
 "accesses": [ ... M ints, tag numbers 1..n ... ]}
```

`left[k-1]` / `right[k-1]` give the initial left/right child **tag** of tag
`k` (`0` = no child); `root` is the initial display root. `accesses[i]` is
the tag the `i`-th visitor requests; `season_weight[i]` is the per-rotation
cost multiplier in effect for visitor `i`.

## Output (stdout, one JSON object)

```json
{"ops": [[...], [...], ...]}
```

`ops` must have exactly `M` entries. `ops[i]` is an **ordered list of tag
keys** to rotate up, one at a time, immediately **before** visitor `i`'s
search. `rotate_up(k)`: tag `k` trades places with its *current* parent via
one standard single BST rotation (the side is implied by which child `k`
currently is) — `k`'s depth drops by exactly one. Splaying a tag all the way
to the root is simply calling `rotate_up` on it repeatedly, once per level of
its current depth. Rotating the current root is a legal no-op that still
costs its season weight. Each key in `ops[i]` must be an integer in `[1, n]`
(no booleans); a wrong list length, a bad key, a crash, a timeout, or
non-JSON output scores that instance `0.0`.

## Cost and scoring (deterministic; minimize)

The evaluator replays your trace itself: for visitor `i` it applies
`ops[i]` (each rotation costs `season_weight[i]`), **then** walks from the
(possibly just-changed) root down to `accesses[i]`, charging `1` per pointer
hop. Your `cost` is the sum of every charge over the whole trace — this is
the hop-vs-restructure trade-off: a rotation you buy now only pays off if it
shortens hops you actually make later.

Per instance, `baseline` is the cost of the policy that **never rotates**
(pure hop cost against the given, unchanged tree). Score:

```
r = min(1, 0.1 * baseline / max(cost, 1e-12))
```

Reproducing "never rotate" scores exactly `0.1`. A policy that restructures
wisely pushes its cost below baseline and its score above `0.1`, toward
(but, on any of these traces, comfortably short of) `1.0`. A policy that
restructures wastefully can push its cost *above* baseline, scoring **below**
`0.1`. Your final score is the mean of `r` over 10 fixed seeded day-traces.

## What to notice

Each trace mixes visitor-behavior regimes that are never announced inline —
tight repeat-visit "regular clientele" runs (a small working set), one-pass
"processions" (strictly increasing tag order, each tag touched once), pacing
"echo-walks" (alternating low/high tags), and raffle-drawn (uniform random)
order — often stitched together, with the season's per-rotation weight
changing between them. "Always splay the requested tag to the root" wins
handsomely during a cheap, repeat-heavy run, but during a peak-season
procession or echo-walk — where no tag repeats, so restructuring buys
nothing — it pays the inflated per-rotation weight on *every single visitor*
for zero long-run benefit, and can finish worse than doing nothing at all.
The winning move is to **meter your own amortized benefit online**: watch
recent-access statistics to estimate whether the current phase has enough
repeat traffic to repay a rotation's cost right now, and only pay when it
does — a phase detector, not a fixed rotate-always-or-never rule.
