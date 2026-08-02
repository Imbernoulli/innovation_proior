# Smallest Lying Coalition

## Problem
An election has `n` honest voters and `m` candidates (indexed `0..m-1`). Voter `i` submits a
strict ranking (best to worst) of all candidates. A **rule** turns the profile into a winner.
Two rule families appear in this task:

- **SCORE** (`rule_type = 0`): a score vector `w[0] >= w[1] >= ... >= w[m-1]`, `w[0] > w[m-1]`,
  is given. Each ballot awards `w[pos]` points to the candidate placed at position `pos`
  (0-indexed, 0 = top). The winner is whoever has the highest total score; ties go to the
  smaller candidate index.
- **RUNOFF2** (`rule_type = 1`): first compute plain Borda scores (position `pos` out of `m`
  gets `m-1-pos` points). The **top two** candidates by Borda score advance to a runoff (ties
  broken toward the smaller index). The runoff winner is whichever of the two is ranked higher
  by more voters in their *original submitted ranking* (a tie goes to the smaller index).

You are told the rule, the full sincere profile, and a **target** candidate `p` who does *not*
win under the sincere profile. A subset `S` of voters (a *coalition*) may replace their ballots
with any strict rankings they like; everyone outside `S` keeps their sincere ballot. Your job is
to output a coalition and its new ballots so that `p` wins under the resulting profile, using as
**few** manipulators as possible.

*(Illustrative FORM only, not a hint about which rule variant is used: think of `w` as tunable
per instance -- plurality (`[1,0,...,0]`), Borda (`[m-1,...,0]`) and veto (`[1,...,1,0]`) are all
valid SCORE instances, and the checker may instead hand you a RUNOFF2 instance.)*

## Input (stdin)
```
m n rule_type target
[w_0 w_1 ... w_{m-1}]        <- present only if rule_type == 0
<n lines, each m integers>   <- voter i's ranking, best to worst
```

## Output (stdout)
```
k
<k lines: voterIndex c_0 c_1 ... c_{m-1}>
```
Print the coalition size `k`, then `k` lines, one per manipulator: the 0-indexed voter and their
new ranking (a permutation of `0..m-1`, best to worst). Voters not listed keep their sincere
ballot.

## Feasibility
- `1 <= k <= n`; the `k` voter indices are distinct integers in `[0, n-1]`.
- each new ranking is a permutation of `0..m-1`.
- re-running the stated rule on the resulting profile must elect `target`.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = 1/k` -- smaller certified coalitions score higher.

## Scoring
Let `B = 1/n`, the value of the checker's own trivial witness: **all** `n` voters join the
coalition and every one submits the identical ballot `[target, then every other candidate in
ascending index order]` (this always elects `target`, under either rule). Then:
```
sc = min(1000.0, 100.0 * F / B)
Ratio = sc / 1000.0
```
Matching the full-electorate witness scores `0.1`; a coalition `10x` smaller caps at `1.0`.

## Constraints
- `3 <= m <= 6`, `8 <= n <= 34`.
- Exactly one of the two rule types is used per test case.
- Time limit 5s, memory 512m.

## Example
Suppose `m=3, n=4, rule_type=0`, `w = [2,1,0]` (Borda), target `p=2`, and the sincere profile is
`[0,1,2] [0,1,2] [1,2,0] [1,2,0]`: totals are 0 -> 4, 1 -> 6, 2 -> 2, so candidate `1` is the
sincere winner. The full-electorate baseline (`k=4`, everyone bullet-votes `2`) scores `0.1`.
If instead voter `2` alone switches from `[1,2,0]` to `[2,0,1]` (output line `2 2 0 1`), totals
become 0 -> 5, 1 -> 4, 2 -> 3: candidate `0` now wins -- flipping one voter changed *who* wins
without helping `2` at all, since `2` still trails both rivals. Reaching `Ratio > 0.1` requires a
coalition (and a ballot template) chosen with the rule's structure in mind, which is exactly what
the harder generated cases test.
