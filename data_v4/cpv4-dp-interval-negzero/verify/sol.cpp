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
