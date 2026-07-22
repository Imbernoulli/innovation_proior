# Banquet Line Antiaffinity: Replica Kitchens for Shared Prep Stations

## Problem

A catering company runs a shared bank of `K` prep kitchens for a whole banquet
season. There are `D` dishes on the season's menus. Every dish must be kept
**prep-ready** (its mise en place stocked) at exactly `R` different kitchens
(its *replica set*) — redundancy so any of those kitchens can plate it on
short notice. Kitchen `k` has a stocking capacity `cap[k]`: the number of
distinct dishes it can keep prepped at once (its total replica load may not
exceed `cap[k]`).

The season contains `B` **courses**: fixed moments where a specific *set* of
dishes must all be plated and go out to the room together (a tasting flight,
a shared banquet round, etc.). For every dish in a course, exactly one of
that dish's `R` replica kitchens must actually plate it. If several dishes of
the same course all end up plated by the same kitchen, that kitchen is the
bottleneck for the whole course: define the course's **makespan** as the
largest number of dishes any single kitchen plates for it. The season's cost
is the sum of these makespans over all `B` courses — a kitchen that is
perfectly balanced in *total* workload across the season can still wreck many
individual courses if its replicas keep landing next to dishes it is
routinely plated alongside.

You must output **both** decisions: (1) each dish's `R`-kitchen replica set
(subject to capacity), and (2) for every course, which of each dish's own
replica kitchens actually serves it. Minimize the summed makespan.

## Input (stdin)

```
K R D B
cap_0 cap_1 ... cap_{K-1}
s_1 d_1 d_2 ... d_{s_1}      (course 1: s_1 dish ids, 0-indexed, distinct)
...
s_B d_1 d_2 ... d_{s_B}      (course B)
```
`1 <= R < K <= 40`, `D` up to ~1100, `B` up to ~350, each course size
`s_c` between 5 and 50. Dish ids are in `[0, D)`.

## Output (stdout)

```
r_{0,1} ... r_{0,R}          (dish 0's R replica kitchens, 0-indexed, distinct)
...
r_{D-1,1} ... r_{D-1,R}      (dish D-1)
k_1 ... k_{s_1}              (course 1: the serving kitchen for each of its
                               dishes, in the SAME order as the input course)
...
k_1 ... k_{s_B}              (course B)
```

## Feasibility

- Each dish's `R` kitchen ids must be **distinct** and in `[0, K)`.
- For every kitchen `k`, the number of dishes whose replica set contains `k`
  must not exceed `cap[k]`.
- For every course and every dish in it, the chosen serving kitchen must be
  **one of that dish's own `R` declared replica kitchens**.
Any violation scores `Ratio: 0.0`.

## Objective & Scoring

Let `F` = sum over courses of (max, over kitchens, of the number of that
course's dishes served by that kitchen) — smaller is better. The checker also
builds its own simple internal baseline `B` (a fixed, frequency-balanced
replica placement with no per-course adaptivity) and reports
`Ratio = min(1000, 100*B/F) / 1000`, so matching that naive baseline scores
about `0.1`, and every point below it in `F` raises the score, capped at
`1.0`.

## Example (illustrative shape only — not real test data)

`K=4, R=2, D=3, B=1`, course 1 = dishes `{0,1,2}`. Suppose dish 0 → kitchens
`{0,1}`, dish 1 → `{0,1}`, dish 2 → `{2,3}`. Then for the single course, dish
0 and dish 1 are forced onto only `{0,1}` (route them to different kitchens,
1 each), and dish 2 goes to kitchen 2 or 3: every kitchen serves at most 1
dish, so the course makespan is 1. Had dish 1 instead been given replicas
`{2,3}` too, kitchen 2 (or 3) might be forced to serve two dishes at once
raising the makespan to 2 — so *which* kitchens a dish's replicas land on,
relative to the dishes it is co-plated with, is what the score is sensitive
to, not just how many total replicas each kitchen holds.

## Constraints

Time limit: 5s. Memory: 512MB. All arithmetic is over small integers;
scoring is exact and deterministic.
