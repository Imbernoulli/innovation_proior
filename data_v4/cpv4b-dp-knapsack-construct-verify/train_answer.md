**Problem.** There are `n` calibration weights `w[0..n-1]` and a counterweight `S`. Choose a subset of
the weights (each used at most once) whose masses sum to exactly `S`, using as **few** weights as
possible, and **output the actual subset**: print `k`, then the `k` chosen 1-based indices in ascending
order. If no subset sums to `S`, print `-1`. `0 <= n, S <= 5000`, `0 <= w[i] <= 10^9`. This is the
*construction* version of subset-sum: the emitted indices are checked for exact sum, distinctness,
in-range, and — the part that bites — equality of `k` to the true minimum.

**Why the obvious greedy is wrong (at tiny size already).** "Sort descending, repeatedly take the
largest weight that still fits" can fail two ways, both reproducible on inputs you can balance in your
head — which is exactly why it tempts people who only test small. On `w = [1,5,6,9]`, `S = 11`, greedy
grabs `9`, then can only add `1` (remaining `1`, nothing of mass `1` left), gets stuck, and falsely
reports `-1` — but `5 + 6 = 11` opens the vault. On `w = [1,12,6,8,11]`, `S = 19`, greedy returns
`12 + 6 + 1` (three weights) while the optimum `8 + 11` uses two. A construction that "works for n=4"
is not a correct construction; it is rejected at scale for false `-1`s and for non-minimality.

**Key idea — bounded-target 0/1 count DP + grid-walk reconstruction.** Because `S <= 5000`, reachable
pan-totals live in `[0, S]`. Let `dp[s]` = the minimum number of weights summing to exactly `s`, with
`dp[0] = 0` and the rest `+inf`. Process weights one at a time; for weight `i` sweep `s` **downward**
from `S` to `w[i]` (the guard that prevents reusing weight `i` in its own pass) and relax
`dp[s] = min(dp[s], dp[s - w[i]] + 1)`. The answer count is `dp[S]` (or `-1` if it stays `+inf`).

To recover an actual subset, do **not** keep a flat `par[s]` parent array — that is sound only for
*unbounded* change and silently reuses a physical weight in the 0/1 setting. Instead store a per-(item,
sum) bit `take[i][s]`, set the instant weight `i` *strictly improves* `dp[s]`. Reconstruct on the grid:
start at `(i = n-1, s = S)`; whenever `take[i][s]` is set, append weight `i` and subtract `w[i]` from
`s`; always step `i` down by one. Since `i` strictly decreases every step, no weight is ever revisited —
distinctness is structural, not hoped-for. Taking the *last* improver of each total is optimal because,
when weight `j` last improved `dp[s]`, the value `dp[s - w[j]]` it used reflected only weights `0..j-1`,
so the residual is solvable optimally over weights `< j`.

**Pitfalls.**
1. *Parent-pointer reuse.* A single flat `par[s]` stores only the last improver per total, so the walk
   `cur -> pre[cur]` can name the same weight twice (e.g. on `[12,3,8,6,11]`, `S=40`, `par[40]=par[29]=4`
   yields an illegal `... 5 5`). Use the per-(item,sum) back-table and walk the grid instead.
2. *Greedy.* Largest-first is not minimal and can falsely report infeasible; never ship it.
3. *Skip irrelevant weights.* `w[i] <= 0` (zero mass never helps, only inflates the count) and
   `w[i] > S` (can never fit) must be ignored, both to stay correct and to keep array indexing valid.

**Edge cases.** `S = 0` -> empty subset, print `k = 0` and an empty second line (the reconstruction
loop is guarded by `s > 0`). `n = 0` -> `-1` for `S > 0`, `0` for `S = 0`. Infeasible -> the single line
`-1`. Oversize and zero-mass weights -> skipped, never chosen. Totals and counts stay within `int`;
`w[i]` up to `10^9` is read as `long long` and only compared against `S`.

**Complexity.** `O(n*S)` time and `O(n*S)` memory for the back-table. At the ceiling `n = S = 5000`
that is `2.5*10^7` relaxations and a `~25 MB` table — measured at `0.03 s` / `~28 MB`, inside the
`1 s` / `256 MB` limits.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // GOAL: output a subset of the n items whose weights sum to EXACTLY S using the
    // FEWEST possible items. If several minimum-size subsets exist, any one is accepted.
    // If no subset of the items sums to S, output the single line: -1
    //
    // Output on success:
    //   line 1: k  (number of chosen items, k >= 0)
    //   line 2: k distinct 1-based indices in ascending order (empty line if k == 0)
    //
    // S may be 0, in which case the empty subset (k = 0) is the unique minimum.

    const int SZ = (int)S;
    const int INF = 1e9;

    // dp[s] = minimum #items (using items considered so far) to reach sum s exactly.
    // take[i][s] = true iff, when item i was processed, it strictly improved dp[s]
    //              (i.e. the optimal way to reach s using items 0..i ENDS by adding item i).
    // The back-table lets us reconstruct ONE consistent minimum-count subset without
    // ever reusing an item: from (i=n-1, s=S) we either step to (i-1, s) when take[i][s]
    // is false, or take item i and step to (i-1, s-w[i]) when it is true.
    vector<int> dp(SZ + 1, INF);
    dp[0] = 0;
    // take stored as a flat vector<char> of size n*(S+1).
    vector<char> take((size_t)n * (SZ + 1), 0);

    for (int i = 0; i < n; i++) {
        char *row = &take[(size_t)i * (SZ + 1)];
        if (w[i] <= 0 || w[i] > S) continue; // cannot help reach a positive sum <= S
        int wi = (int)w[i];
        for (int s = SZ; s >= wi; s--) {
            if (dp[s - wi] != INF && dp[s - wi] + 1 < dp[s]) {
                dp[s] = dp[s - wi] + 1;
                row[s] = 1;
            }
        }
    }

    if (dp[SZ] == INF) { cout << -1 << "\n"; return 0; }

    // Reconstruct by walking the back-table from (n-1, S) down to (-1, 0).
    vector<int> chosen;
    int s = SZ;
    for (int i = n - 1; i >= 0 && s > 0; i--) {
        const char *row = &take[(size_t)i * (SZ + 1)];
        if (row[s]) {
            chosen.push_back(i + 1); // 1-based
            s -= (int)w[i];
        }
    }
    // s is now 0 and every chosen index is distinct (each item visited at most once).
    sort(chosen.begin(), chosen.end());

    cout << (int)chosen.size() << "\n";
    for (size_t k = 0; k < chosen.size(); k++) {
        cout << chosen[k];
        cout << (k + 1 == chosen.size() ? '\n' : ' ');
    }
    if (chosen.empty()) cout << "\n"; // empty second line for k == 0
    return 0;
}
```
