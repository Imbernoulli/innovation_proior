#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;          // no input -> nothing to do
    vector<long long> p(n);
    for (auto &x : p) cin >> x;

    sort(p.begin(), p.end());                // positions need not arrive sorted

    // feasible(d): can we place all k items on the sorted positions so that
    // every pair of chosen positions differs by at least d? Greedy: always
    // anchor at the first position, then keep the next position that is at
    // least d beyond the last one we kept. The count this greedy achieves is
    // the MAXIMUM number of items placeable with min-gap >= d.
    auto feasible = [&](long long d) -> bool {
        long long placed = 1;                // first item at p[0]
        long long last = p[0];
        for (int i = 1; i < n && placed < k; i++) {
            if (p[i] - last >= d) {
                placed++;
                last = p[i];
            }
        }
        return placed >= k;
    };

    // k == 1: a single item has no pair, so the min-gap is undefined; by
    // convention we report 0 (no constraint to satisfy).
    if (k <= 1) {
        cout << 0 << "\n";
        return 0;
    }

    // Binary search the largest d for which feasible(d) is true. feasible is
    // monotone: if we can achieve min-gap >= d, we can achieve >= d' for any
    // d' <= d. Search d in [0, span], where span = p[n-1] - p[0] is the
    // largest gap any pair can have.
    long long lo = 0, hi = p[n - 1] - p[0];
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;   // upper mid: avoids infinite loop
        if (feasible(mid)) lo = mid;
        else hi = mid - 1;
    }

    cout << lo << "\n";
    return 0;
}
