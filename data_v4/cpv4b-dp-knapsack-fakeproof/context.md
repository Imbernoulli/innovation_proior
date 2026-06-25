# Resonant Knapsack: counting subsets by mass and phase

## Research question

A field technician is tuning a crystal resonator. There are `n` crystal shards on the bench. Shard
`i` has an integer **mass** `w[i]` and an integer **phase** `p[i]`. The technician welds a subset `S`
of the shards into a single resonator (the empty subset is allowed: that is the bare mount). The
welded resonator **resonates** at the target tuning `(W, q)` exactly when

- the total mass equals the target mass: `sum_{i in S} w[i] == W`, and
- the total phase, taken modulo `M`, equals the target residue: `(sum_{i in S} p[i]) mod M == q`.

Both conditions must hold simultaneously. Count **how many distinct subsets** resonate at `(W, q)`,
and report that count modulo `1_000_000_007`. Two subsets are distinct if they differ in at least one
shard, even if they have identical masses and phases — shards are physical objects, not values.

This is a two-constraint counting knapsack: an exact-weight count (the mass dimension) coupled to a
modular count (the phase dimension). The danger is that the two dimensions *look* separable or *look*
uniformly distributed, and a tempting closed form for the phase dimension is easy to assert and wrong.

## Input / output contract

- Input (stdin): the first line has four integers `n W M q`.
  - `0 <= n <= 200`
  - `0 <= W <= 2000`
  - `1 <= M <= 200`
  - `0 <= q < M`
- Then `n` lines follow; line `i` has two integers `w[i] p[i]` with
  - `1 <= w[i] <= 2000`
  - `0 <= p[i] <= 10^9`  (phases may be far larger than `M`; only `p[i] mod M` matters).
- Output (stdout): a single line with the number of resonant subsets, modulo `1_000_000_007`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 4`, `W = 5`, `M = 3`, `q = 2`, and shards
`(w, p) = (2, 1), (3, 2), (2, 0), (1, 1)`, the answer is `2`. The two resonant subsets are
`{shard 1, shard 2}` (mass `3 + 2 = 5`, phase `2 + 0 = 2 ≡ 2`) and
`{shard 0, shard 2, shard 3}` (mass `2 + 2 + 1 = 5`, phase `1 + 0 + 1 = 2 ≡ 2`).

## Background

This sits in the 0/1 knapsack family: each shard is taken at most once, and we are *counting*
configurations rather than maximizing a value. Two families of approach are on the table before
committing:

- **Factor the phase dimension out with a closed form.** It is tempting to first count the subsets
  that hit mass exactly `W` (a one-dimensional exact-subset-sum count), then argue that those subsets
  spread across the `M` phase residues in some clean, assertable way — for instance uniformly, giving
  `(count at mass W) / M` per residue. If true this would collapse the problem to a single dimension.
  The open question is whether any such closed form for the residue distribution is actually valid.
- **Carry both constraints in one DP table.** Track, for every reachable mass and every phase residue,
  how many subsets realize that exact `(mass, residue)` pair, and read off the target cell at the end.
  This is a direct product-of-dimensions table. The open questions are the exact transition, the
  iteration order that keeps each shard 0/1, and whether the table fits the limits.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (only the empty subset; resonant iff `W = 0` and `q = 0`);
`M = 1` (the phase constraint is vacuous, `q` is always `0`); shards with equal masses; shards with
equal phases; shards whose mass exceeds `W` (they can never be welded in); phases far larger than `M`
(must be reduced mod `M`); targets `q` for which the count is `0`; and full-size cases
`n = 200, W = 2000, M = 200` to pin down both the time budget and the modular arithmetic.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long W;
    int M, q;
    if (!(cin >> n >> W >> M >> q)) return 0;

    vector<long long> w(n);
    vector<int> p(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> p[i];

    // TODO: count subsets with total mass == W and total phase (mod M) == q,
    //       modulo 1e9+7 (empty subset allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
