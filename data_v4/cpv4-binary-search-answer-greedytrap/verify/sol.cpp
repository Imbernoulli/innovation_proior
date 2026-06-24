#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Feasibility: can we partition a[0..n-1] into at most k contiguous blocks,
    // each with block-sum <= T?  Greedy: extend the current block while the running
    // sum stays <= T; when it would exceed, cut (start a new block at this bed).
    auto feasible = [&](long long T) -> bool {
        long long cur = 0;
        long long blocks = 1;            // we always use at least one block
        for (int i = 0; i < n; i++) {
            if (a[i] > T) return false;  // a single bed already exceeds T
            if (cur + a[i] > T) {        // start a new block at bed i
                blocks++;
                cur = a[i];
                if (blocks > k) return false;
            } else {
                cur += a[i];
            }
        }
        return blocks <= k;
    };

    // Lower bound: the largest single bed (some block must hold it).
    // Upper bound: the total (one block holds everything).
    long long lo = 0, hi = 0;
    for (long long x : a) { lo = max(lo, x); hi += x; }

    // Empty greenhouse: no beds, finishing time is 0.
    if (n == 0) { cout << 0 << "\n"; return 0; }
    // If k >= n every bed can be its own block, so the answer is the max bed.
    // (The binary search handles this too, but it is a clean sanity anchor.)

    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
