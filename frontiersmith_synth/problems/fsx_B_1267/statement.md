# Claim Ring Audit: Investigating Groups, Not Claims

## Problem

An insurance book has **N** claims. Every claim links exactly one
**claimant**, one **provider**, and one **adjuster** (integer ids, each
role has its own id space) and carries a declared **amount**, a public
**plausibility score** in `[0,1]` (higher = looks more routine — it is
computed upstream from things like documentation completeness and timing,
and is already given to you per claim), and an integer **investigation
cost**. You have a total investigation **budget M**: you may pick any set
of claims to audit as long as the sum of their costs does not exceed `M`.

Some claims are fraudulent. A few are lone, clumsy fraud: on their own
they simply look wrong (their plausibility score is honestly low). But
most of the hidden fraudulent value sits inside **collusion rings**: a
small, fixed cluster of parties (say two claimants, two providers, one
adjuster) who file many claims *among themselves*, always drawing from
that same tiny cluster. Every individual claim in a ring is written
carefully enough to score a plausibility on par with ordinary business —
nothing about any single ring claim looks wrong. The only signature a
ring leaves is structural: the same (claimant, provider) / (claimant,
adjuster) / (provider, adjuster) party *pair* keeps recurring across
several different claims, because the ring only ever draws from its own
small pool of colluders, while ordinary claims and lone fraud draw their
three parties from a much wider pool and essentially never repeat a pair
by chance. Auditing one claim in a ring at random teaches you almost
nothing about that claim in isolation — but the recurring-pair pattern
across claims tells you exactly where to look.

Your job: pick which claims to audit under the budget to recover as much
true fraudulent value as possible.

## Input (stdin)

```
N M NC NP NA testId
claimant_1 provider_1 adjuster_1 amount_1 plausibility_1 cost_1
...
claimant_N provider_N adjuster_N amount_N plausibility_N cost_N
```
`N` claims, budget `M`, and the total id-space sizes `NC`/`NP`/`NA` for
claimant/provider/adjuster ids (each `0 <= id < NC/NP/NA` respectively).
`testId` is an opaque ladder index. Claim order carries no information.

## Output (stdout)

```
K
i_1 i_2 ... i_K
```
`K` is how many claims you choose to audit (`0 <= K <= N`), followed by
their `K` distinct 0-indexed claim ids (any order). The sum of the costs
of the chosen claims must not exceed `M`.

## Feasibility

Your output is rejected (score 0) if: `K` is outside `[0,N]`; any index is
non-integer, out of `[0,N)`, or repeated; or the total investigation cost
of your chosen claims exceeds `M`.

## Objective

Auditing a claim reveals the truth about that claim only. Your recovered
value is the sum of the declared `amount` over exactly the claims you
chose that are truly fraudulent (fraudulent claims you did not choose,
and non-fraudulent claims you did choose, contribute nothing).

## Scoring

The checker also computes a baseline recovered value `B`: the value
recovered by auditing claims **in the order they are listed in the
input**, greedily filling the same budget `M` (claim order carries no
information — it was shuffled before being printed — so this baseline is
structure-blind, plausibility-blind, and amount-blind). Your ratio is
`min(1000, 100 * F / B) / 1000`, where `F` is your recovered value,
printed as `Ratio: <value>`. A budget spent with no exploitable signal
scores near 0.1; the ratio never exceeds 1.0.

## Constraints

`40 <= N <= 320`, costs are small positive integers, amounts are in
`[50, 650]`. Time limit 5s, memory 512MB.

## Example (worked score, illustrative shape only)

Suppose `N=4`, `M=2`, and claims 0,1 share adjuster `7` with providers
`3` and `4` respectively while claimant `9` appears on both (so the pair
`(claimant=9, adjuster=7)` recurs) — claims 2,3 involve entirely distinct,
unrepeated parties. If claims 0 and 1 are truly the fraudulent ring and
you spend your budget of 2 auditing exactly those two (say amounts 200
and 150), `F = 350`. If instead you had picked claims 2 and 3, unaware of
the recurring pair, and both were innocent, `F = 0`. This illustrates the
shape of the recurring-pair signal only — it is not the actual test data.
