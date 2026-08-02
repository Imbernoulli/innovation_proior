# Shard Assignment Under Skew, Cross-Shard Cost, and Resharding Stability

## Problem
A key-value store has `N` keys spread across `K` shards. You are given, for a
fresh resharding decision: each key's storage weight, the shard each key
lived in *before* this resharding (or "new key" if it did not previously
exist), and a weighted trace of transactions that touched pairs of keys
together. You must output a NEW shard for every key.

Three effects determine the cost of your assignment, and they pull in
different directions:

- **Access skew**: shards with very different total weight are bad (hot/
  overloaded shards). Penalized by how far each shard's load is from the
  average, squared.
- **Cross-shard transaction cost**: every recorded transaction between two
  keys costs extra if the keys end up on different shards (it must be
  coordinated across the network); free if they land on the same shard.
- **Resharding stability**: moving a key away from where it lived before
  this resharding costs extra, proportional to the key's weight (data must
  be migrated).

The single most uniformly load-distributing assignment (e.g. spreading keys
by hash) minimizes the skew term — but it has no notion of which keys
transact together, so it tends to maximize the cross-shard term. The
cheapest overall assignment often *accepts* some size imbalance in order to
keep frequently-co-transacting keys together, and avoids needless migration
when the previous placement was already good.

## Input (stdin)
```
N K
A B G
w_1 w_2 ... w_N
p_1 p_2 ... p_N
M
u_1 v_1 c_1
...
u_M v_M c_M
```
`N` keys (indices `0..N-1`), `K` shards (indices `0..K-1`). `A`, `B`, `G`
are positive cost coefficients for the skew, cross-shard, and migration
terms respectively. `w_i` (positive integer) is key `i`'s weight. `p_i` is
key `i`'s previous shard (`0..K-1`), or `-1` if key `i` is brand new (no
migration cost either way). `M` is the number of transaction records; each
gives an unordered pair of DISTINCT keys `u_i, v_i` and a positive integer
weight `c_i` (a pair appears at most once, already aggregated).

## Output (stdout)
Exactly `N` integers (any whitespace-separated layout), token `i` (0-indexed)
giving key `i`'s new shard, each in `[0, K)`.

## Feasibility
The output must contain exactly `N` tokens, each a base-10 integer in
`[0, K)`. Any wrong count, non-integer token, or out-of-range value makes
the whole answer infeasible (score 0).

## Objective (what the score rewards)
Let `assign[i]` be your chosen shard for key `i`, `load_s` the sum of
weights of keys assigned to shard `s`, and `avg = (sum of all weights)/K`.
```
skew      = sum_s (load_s - avg)^2
cross     = sum over (u,v,c) with assign[u] != assign[v] of c
migration = sum over i with p_i != -1 and assign[i] != p_i of w_i
F = A*skew + B*cross + G*migration
```
Lower `F` is better.

## Scoring
The checker computes `F` for your assignment and `F_ref` = the cost of the
round-robin assignment `assign[i] = i % K` (its own reference construction,
which ignores transactions and prior placement), then reports
`Ratio = min(1000, 100*F_ref/F) / 1000` (lower `F` ⇒ higher ratio; matching
`F_ref` gives ≈0.1). Your score is the mean ratio across 10 hidden test
instances.

## Constraints
`2 ≤ N ≤ 40`, `2 ≤ K ≤ 6`, `0 ≤ M ≤ 3N`, `1 ≤ w_i ≤ 25`,
`1 ≤ A,B,G ≤ 30` (integers). Time limit 5s.

## Example (illustrative FORM only — not a real hidden case)
`N=4, K=2`, `A=1 B=10 G=0`, weights `1 1 1 1`, all `p_i=-1`, one transaction
`(0,1,c=20)`. Round-robin gives shards `[0,1,0,1]`: skew=0, keys 0,1 land on
different shards so cross pays `20` ⇒ `F_ref = 10*0 + 10*20 = 200`.
Co-locating the transacting pair, e.g. shards `[0,0,1,1]`: skew=0 (loads
2,2), no cut edge ⇒ `F = 0`. Here co-location wins outright; with denser,
larger transaction graphs the trade-off against skew becomes real, and the
best choice depends on the actual weights and coefficients in the input.
