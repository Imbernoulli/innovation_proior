#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // We must pick a contiguous block of length >= k. Its "robust floor" is the
    // minimum element of the block. Maximize that floor over all valid blocks.
    // If no block of length >= k exists (n < k), report INFEASIBLE.
    if ((long long)n < k) {
        cout << "INFEASIBLE" << "\n";
        return 0;
    }

    // Binary search on the answer x: feasible(x) = exists a run of >= k
    // consecutive positions all with a[i] >= x. feasible is monotone:
    // raising x can only shorten runs, so feasibility is non-increasing in x.
    // Bounds: any single element is >= min(a); a window's min is <= max(a).
    long long lo = LLONG_MAX, hi = LLONG_MIN;
    for (long long v : a) { lo = min(lo, v); hi = max(hi, v); }
    // lo is achievable (whole-array window of length n>=k has min >= lo),
    // hi+1 is not (no element is >= hi+1). Binary search the largest feasible x.

    auto feasible = [&](long long x) -> bool {
        long long run = 0;
        for (long long v : a) {
            if (v >= x) { run++; if (run >= k) return true; }
            else run = 0;
        }
        return false;
    };

    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2; // upper mid to avoid infinite loop
        if (feasible(mid)) lo = mid;
        else hi = mid - 1;
    }

    cout << lo << "\n";
    return 0;
}
