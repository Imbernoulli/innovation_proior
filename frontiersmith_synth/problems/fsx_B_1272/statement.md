# Coverage-Constrained Alert Triage

## Problem
A small anti-money-laundering team must decide which of today's alerts to actually
investigate. Each alert belongs to one of `K` **typologies** (patterns of suspicious
activity), carries a **model risk score** in `[0,100]` (the detection model's estimate of
how suspicious it is), and costs a fixed number of **investigator-minutes** to work. The
team has a total time budget for the shift.

Investigating an alert recovers a monetary amount that depends on the alert's typology and
score through relationships the team does not get to see directly — **not disclosed in the
input** (real triage teams do not know exactly what an alert is worth before working it;
they only see the visible features). The visible score is a useful but IMPERFECT signal:
it is not equally informative for every typology, and a typology's *typical* score level
says nothing about how much money is actually at stake once you do investigate it.

On top of the budget, the regulator publishes a **minimum coverage requirement** per
typology for this shift: the team must investigate at least that many alerts of that
typology, regardless of how promising their scores look, or the whole work plan is
rejected outright (no partial credit — regulatory coverage is not negotiable).

## Input (stdin)
```
testId N K C
N lines: id typology cost score       (1<=id<=N, 1<=typology<=K, cost>=1, 0<=score<=100)
K lines: typology min_cover           (minimum number of that typology's alerts to work)
```

## Output (stdout)
```
k
k lines or a single line of k alert ids: the ids to investigate (any whitespace-separated
layout is accepted)
```

## Feasibility
A work plan is valid iff **all** hold:
- `0 <= k <= N`, and exactly `k` ids follow;
- every id is a valid alert id (`1..N`), each id finite and appearing at most once;
- the sum of `cost` over chosen ids does not exceed `C`;
- for every typology `t`, the number of chosen alerts with that typology is `>= min_cover[t]`.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F`, the total recovered value summed over the chosen alerts. Each alert's
recovered value is a positive, typology- and score-dependent amount determined by the
checker (not visible to the solver) — score helps within a typology, but the relationship
between score and recovered value is typology-specific and not uniform across typologies.

## Scoring
The checker also builds its own reference plan `B`: for every typology, take its
`min_cover[t]` cheapest alerts (to satisfy the floor as cheaply as possible), then fill any
remaining budget by scanning alert ids in ascending order. `B` is `B`'s recovered value
(floored at `1.0`). With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the reference plan scores `Ratio = 0.1`; a plan recovering `10x` its value caps at
`1.0`.

## Constraints
- `K = 5`, `31 <= N <= 94`, alerts and floors fit easily in memory.
- The instance always admits SOME plan meeting every floor within budget `C`.
- Time limit 5s, memory 512m.

## Example (illustrative shape only, not the scoring formula's constants)
Suppose two alerts, both typology 1, `min_cover[1] = 1`, costs `10` and `10`, `C = 15`.
The reference plan can only afford one of them to satisfy the floor, so `B` is that single
alert's recovered value. A plan that instead picks the id that turns out to carry the
higher recovered value scores above `0.1`; a plan investigating neither alert (or
exceeding `C`, or missing a typology's floor) scores `0.0`.
