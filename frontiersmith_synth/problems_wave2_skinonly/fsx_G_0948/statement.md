# Campus Mental Health Program Portfolio

## Problem
You are assembling a portfolio of candidate assets for **campus mental health program portfolio**. Each asset has a cost, group,
base value, lattice coordinates, and a bit mask describing capabilities. Select a feasible subset
that maximizes the deterministic utility used by the checker.

This is a constructive optimization problem in the `student-mental-health-program-mix` family: many feasible subsets are
accepted, but stronger submissions balance budget, group diversity, feature coverage, and hidden
pairwise compatibility better than simple cost-first choices.

## Input
The input is read from stdin.

The first line contains `n m g budget groupCap salt profile`.
Each of the next `n` lines contains `cost group value x y mask`.

`mask` is a non-negative decimal integer whose binary bits encode covered capabilities.

## Output
Print `q`, the number of selected assets, followed by exactly `q` distinct 1-indexed asset ids.

## Feasibility
The selected ids must be in range and distinct. Total cost must not exceed `budget`; no group may
appear more than `groupCap` times; and the checker rejects deterministic conflict pairs based on
coordinates, groups, feature overlap, `salt`, and `profile`.

## Objective and Scoring
The checker scores the selected set by summing base values, saturated capability coverage bonuses,
group-diversity bonuses, and pairwise cross-group synergy bonuses. Let `F` be your feasible score and
let `B` be the checker's internal cost-first baseline score. The reported ratio is
`min(1, 0.1 * F / B)`. The baseline construction therefore scores about `0.1`; better constructions
receive higher ratios.

## Constraints
There are 10 deterministic generator cases. Larger cases have more assets and denser conflicts.
Time limit: 5 seconds. Memory limit: 512 MB.

## Example
For a tiny instance, the cost-first baseline might choose low-cost assets `1 4 7` and score `B=300`.
If your feasible subset scores `F=540`, the checker prints `Ratio: 0.180000`.
