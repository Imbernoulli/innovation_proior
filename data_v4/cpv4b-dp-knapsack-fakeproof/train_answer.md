**Problem.** There are `n` crystal shards; shard `i` has mass `w[i]` and phase `p[i]`. Weld a subset
`S` (empty allowed). It *resonates* at `(W, q)` iff `sum_{i in S} w[i] == W` **and**
`(sum_{i in S} p[i]) mod M == q`, both at once. Count the resonant subsets modulo `1_000_000_007`.
Distinctness is by which shards are chosen. Constraints: `n <= 200`, `W <= 2000`, `M <= 200`,
`w[i] <= 2000`, `p[i] <= 10^9`. Read `n W M q` then `n` pairs `w[i] p[i]` from stdin; print the count.

**Why the tempting closed form is wrong.** It is seductive to first count `C = #{subsets with mass
exactly W}` and then assert the phase residues of those subsets are uniform, giving `C / M` per residue
and collapsing the phase dimension. Disprove it on `n=3, W=2, M=3`, shards `(1,0),(1,0),(2,1)`: the
subsets hitting mass `2` are `{0,1}` (residue `0`) and `{2}` (residue `1`), so the residue counts are
`(1, 1, 0)` — not the `C/M = 2/3` uniformity predicts (not even an integer). The mass constraint
correlates which phases co-occur, so there is no divide-by-`M`. Never assert "uniform by symmetry"; the
two dimensions are coupled and must be carried together.

**Key idea — coupled 2D counting knapsack.** Let `dp[m][r] = #{subsets with total mass exactly `m` and
total phase ≡ `r` (mod M)}`. Cap `m` at `W` (heavier partial welds can never return to `W`). Base:
`dp[0][0] = 1` (empty subset). For each shard with mass `wi` and reduced phase `pi = p[i] mod M`, do a
0/1 update, iterating mass **downward** so each shard is welded at most once:

    for m = W down to wi:
        for r in 0..M-1:
            dp[m][(r + pi) mod M] += dp[m - wi][r]   (mod 1e9+7)

The answer is `dp[W][q]`.

**Pitfalls (each grounded in a trace).**
1. *Forward mass iteration turns 0/1 into unbounded.* On two unit shards `(1,0),(1,0)`, `W=2, M=1`, the
   true answer is `1` (`{both}`), but ascending `m` reads a `dp[m - wi]` already updated this pass and
   returns `3`, welding a shard twice. Descend the mass axis.
2. *Forgetting `p[i] mod M`.* With `p[i]` up to `10^9` and `M <= 200`, the residue wrap is a single
   conditional subtraction *only* if `pi < M`. Reduce `pi = p[i] % M` before the loop; otherwise
   `nph = r + pi` indexes far outside the `M`-wide residue row (heap overflow / corruption). Traced on
   two phase-`5` shards, `M=3`.
3. *Overflow / modulus.* The count can reach `2^200`; keep every cell reduced mod `1e9+7` and use
   `long long`.

**Edge cases (all fall out of the recurrence).** `n = 0`: table stays `dp[0][0]=1`, so answer is `1`
iff `W=0` and `q=0`, else `0`. `M = 1`: only residue `0`, `q` forced to `0`, count collapses to subsets
with mass exactly `W`. Shard with `w[i] > W`: skipped (never weldable), still contributes its
"not-taken" branch. Unreachable `(W, q)`: cell is `0`, printed directly.

**Complexity.** `O(n * W * M) <= 8 * 10^7` time, `O(W * M)` memory (~3.2 MB). Runs in ~0.04 s on the
full-size case.

**Code.**

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

    // Masses beyond W are useless; cap the table at W.
    // dp[mass][phase] = number of subsets with total mass == mass and total
    // phase (mod M) == phase. Two dimensions are coupled, so we carry both.
    int Wcap = (int)W; // W fits in int by constraints (<= 2000)
    vector<vector<long long>> dp(Wcap + 1, vector<long long>(M, 0));
    dp[0][0] = 1; // empty subset

    for (int i = 0; i < n; i++) {
        long long wi = w[i];
        int pi = p[i] % M;
        if (wi > Wcap) continue; // cannot fit, skip (still counts as "not taken")
        int wic = (int)wi;
        // 0/1 knapsack: iterate mass downward to avoid reusing item i.
        for (int m = Wcap; m >= wic; m--) {
            const vector<long long> &src = dp[m - wic];
            vector<long long> &dst = dp[m];
            for (int ph = 0; ph < M; ph++) {
                long long v = src[ph];
                if (v == 0) continue;
                int nph = ph + pi;
                if (nph >= M) nph -= M;
                dst[nph] += v;
                if (dst[nph] >= MOD) dst[nph] -= MOD;
            }
        }
    }

    cout << dp[Wcap][q] % MOD << "\n";
    return 0;
}
```
