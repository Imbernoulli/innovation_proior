**Problem.** A row of `n` crystals has integer charges `c[0..n-1]` (negative, zero, or positive). A fusion welds two *adjacent* clusters of charges `L` and `R` into one cluster of charge `L + R` and pays reward `L * R`. Starting from each crystal alone, perform any sequence of adjacent fusions in any order and stop whenever you like (possibly never). Read `n` and the charges from stdin; print the maximum total reward. Doing nothing scores `0`, so the answer is at least `0`.

**Key structural fact.** Fusions only combine adjacent clusters, and adjacency of contiguous runs is preserved, so every reachable state is a partition of the row into contiguous blocks. A final configuration is therefore: choose a partition into contiguous blocks, fully fuse each block into one cluster, and leave singletons untouched (they pay nothing). Total reward is additive across blocks. This splits the problem into (1) best reward to fully fuse one contiguous range, and (2) which ranges to fuse.

**Key idea — interval DP + partition DP.**

- Inner interval DP. `mergeAll[i][j]` = max total reward to fuse crystals `i..j` into one cluster. The *last* weld joins `[i..k]` and `[k+1..j]`, paying `sum(i..k) * sum(k+1..j)`:
  `mergeAll[i][j] = max_{i<=k<j} ( mergeAll[i][k] + mergeAll[k+1][j] + sum(i..k) * sum(k+1..j) )`,
  base `mergeAll[i][i] = 0` (a single crystal has no weld). Contiguous sums via prefix sums.
- Outer partition DP. `best[p]` = max reward over the first `p` crystals. The last crystal lies in block `[q-1 .. p-1]`:
  `best[p] = max_{1<=q<=p} ( best[q-1] + mergeAll[q-1][p-1] )`, base `best[0] = 0`.
  The case `q = p` uses `mergeAll[p-1][p-1] = 0`, i.e. "leave this crystal alone." Answer is `best[n]`.

**Correctness.** The inner recurrence enumerates the position of the last weld, which uniquely determines the two independent sub-blocks fused beforehand; maximizing over all `k` and over the optimal sub-block rewards gives the optimum for the range (standard matrix-chain/stone-merge argument). The block rewards are independent across a partition because no weld crosses a block boundary, so the partition DP correctly sums them. Taking every crystal as its own singleton yields `0`, so `best[n] >= 0` automatically — no separate `max(...,0)` is needed once the singleton transition is present.

**Pitfalls.**
1. *Sign / "all-negative ⇒ 0" trap.* A negative times a negative is positive, so fusing negative clusters *pays*. The all-negative input usually has a large positive answer (e.g. `[-2,-3,-1] -> 11`), not `0`. The DP never special-cases sign; it just takes products. Defaulting to `0` on all-negative rows is wrong.
2. *Base case of the inner table.* A length-1 range must score `0` (no weld). Leaving the diagonal at a sentinel `-inf` corrupts every block reward that reads it (e.g. `[2,3]` would return garbage instead of `6`). Fix: initialize the table diagonal to `0`.
3. *Don't clamp negative blocks.* The *local* per-cell accumulator must start at `-inf`, not `0`, or a genuinely losing block like `[3,-3] = -9` gets wrongly clamped to `0`. The partition layer needs the true (possibly negative) `mergeAll` so it can choose to split instead.
4. *Overflow.* With `n <= 400` and `|c[i]| <= 10^4`, cluster sums reach `4*10^6` and rewards `~1.6*10^13`; totals exceed 32 bits. Use `long long`. An `int` is a silent wrong-answer.

**Edge cases.** `n = 0` -> `0` (empty schedule). Single crystal -> `0`. All zeros -> `0`. All negatives -> positive (fuse everything). Mixed signs where every fusion loses (e.g. `[-3,4,-3]`) -> `0` (leave all singletons). Two negatives `[-4,-5]` -> `20`.

**Complexity.** Inner table `O(n^3)` time, `O(n^2)` memory; partition `O(n^2)`. For `n = 400` that is `~6.4*10^7` steps and `~1.3 MB`, well within 1 s / 256 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // prefix sums so a contiguous charge sum is O(1)
    vector<long long> pre(n + 1, 0);
    for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + c[i];
    auto rangeSum = [&](int i, int j) { return pre[j + 1] - pre[i]; }; // sum c[i..j]

    const long long NEG = LLONG_MIN / 4;

    // mergeAll[i][j] = max total reward to fuse crystals i..j into ONE cluster.
    // Last fusion joins clusters [i..k] and [k+1..j]; reward of that fusion is
    // (sum i..k) * (sum k+1..j). Base mergeAll[i][i] = 0 (single crystal, no fusion).
    vector<vector<long long>> mergeAll(n, vector<long long>(n, 0));
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            long long best = NEG;
            for (int k = i; k < j; k++) {
                long long left = rangeSum(i, k);
                long long right = rangeSum(k + 1, j);
                long long cand = mergeAll[i][k] + mergeAll[k + 1][j] + left * right;
                if (cand > best) best = cand;
            }
            mergeAll[i][j] = best;
        }
    }

    // best[p] = max reward considering the first p crystals, partitioned into
    // contiguous blocks; each block is fully fused; a block contributes its
    // mergeAll value. A length-1 block contributes 0 (no fusion). best[0] = 0.
    // We may leave crystals unmerged, so the empty action gives 0 overall.
    vector<long long> best(n + 1, 0);
    for (int p = 1; p <= n; p++) {
        long long b = best[p - 1];                 // last crystal alone (block size 1, reward 0)
        for (int q = 1; q <= p; q++) {             // last block = crystals (q-1 .. p-1)
            long long cand = best[q - 1] + mergeAll[q - 1][p - 1];
            if (cand > b) b = cand;
        }
        best[p] = b;
    }

    cout << best[n] << "\n";
    return 0;
}
```
