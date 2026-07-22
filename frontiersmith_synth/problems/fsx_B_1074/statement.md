# Guild Clearinghouse: Minimal-Transfer Settlement

## Problem
The Adventurers' Guild just closed out a season. Over the season, members
lent and borrowed favors from each other constantly, recorded as a long list
of raw IOUs: "party `d` owes party `c` amount `a`". The same pair may owe
each other several times over the season, in either direction. At season's
end the clearinghouse must settle everything with a small, explicit list of
cash transfers.

First **net** the IOUs: party `i`'s net position `balance[i]` is the total
it is owed minus the total it owes, summed over every IOU in the season. A
party with `balance[i] > 0` is a net creditor (it is owed money overall); a
party with `balance[i] < 0` is a net debtor. The sum of all balances is
always exactly zero.

You must then submit a **settlement plan**: an explicit list of transfers
`(u, v, amount)`, each meaning "party `u` pays party `v` amount `amount`".
After applying every transfer, every party's (money received) minus (money
paid) must equal exactly its net balance -- the whole guild is settled.

## Input (stdin)
```
n m
```
then `m` lines, each `d c a`: party `d` (1-indexed, `1..n`) owed party `c` a
total of `a` (a positive integer, `d != c`) at some point during the season.
The same ordered pair `(d, c)` may repeat.

## Output (stdout)
```
k
```
then `k` lines, each `u v a`: party `u` pays party `v` amount `a` (a
positive integer, `u != v`). `k` may be `0` only if every party's net
balance is already zero (not the case in any test here).

## Feasibility
Compute `balance[i]` by netting every IOU in the input. Your submitted
transfers are feasible iff, for every party `i`, (total received by `i`)
minus (total paid by `i`) equals `balance[i]` **exactly**. Any parse error,
out-of-range party id, non-positive/non-finite amount, self-transfer
(`u == v`), wrong token count, or an unsettled party scores `0`.

## Objective
Minimize `k`, the number of settlement transfers.

## Scoring
The checker builds its own trivial feasible settlement: walk parties
`1..n` in order, carrying the running net-balance total forward and
emitting one transfer to the next party whenever that carry is nonzero.
Let `B` be that construction's transfer count. With your feasible
transfer count `k`:
```
Ratio = min(1, 0.1 * B / k)
```
Fewer transfers than the naive chain raises the ratio above `0.1`; a
tenth of `B` caps it at `1.0`.

## Why fewer transfers are possible
Split the `n` parties into any partition of disjoint groups such that each
group's balances sum to exactly zero. Every group of size `s` can be
internally settled with exactly `s - 1` transfers (a chain within the
group), and no cross-group transfer is ever required. So the minimum
possible `k` for a given partition is `n - (number of groups)`, and the true
minimum over *all* valid partitions is `n - (maximum number of disjoint
zero-sum subsets of the balance vector)`. A plan that only ever pays the
current largest debtor into the current largest creditor never goes looking
for these exact subset matches, so it typically finds far fewer groups than
exist.

## Constraints
- `2 <= n <= 500`, `0 <= m <= 6000`.
- `1 <= d, c <= n`, `d != c`, `1 <= a <= 10^6` per IOU.
- `1 <= a <= 10^15` per submitted transfer amount; `0 <= k <= 20000`.
- Deterministic exact-integer scoring; no timing.

## Example
Suppose netting yields `balance = [+7, -7, +3, -2, -1]` for `n=5`. The
naive ascending-order chain needs `B=4` transfers. But `{1,2}` (`+7,-7`)
and `{3,4,5}` (`+3,-2,-1`) are both zero-sum, so `k=1+2=3` transfers
suffice: e.g. `2 1 7`, `4 3 2`, `5 3 1`. That plan scores
`min(1, 0.1*4/3) = 0.133`.
