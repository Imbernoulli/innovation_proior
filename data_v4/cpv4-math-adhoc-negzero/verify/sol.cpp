#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> empty subarray, product 1
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // The empty subarray is allowed and its product is defined as 1, so the answer
    // is at least 1. For a NON-EMPTY subarray ending exactly at position i we track
    // BOTH the largest product (curMax) and the smallest product (curMin), because a
    // negative a[i] turns the most-negative running product into the most-positive one.
    long long best = 1;                    // empty subarray: product 1, always available
    long long curMax = 0, curMin = 0;      // products of best/worst NON-EMPTY run ending at i-1
    bool haveRun = false;                  // does such a non-empty run ending at i-1 exist yet?

    for (int i = 0; i < n; i++) {
        long long x = a[i];
        long long nMax, nMin;
        if (!haveRun) {
            // nothing to extend: the only non-empty window ending at i is {x}
            nMax = x;
            nMin = x;
        } else {
            long long e1 = curMax * x;     // extend the best run by x
            long long e2 = curMin * x;     // extend the worst run by x (matters when x < 0)
            nMax = max({x, e1, e2});       // start fresh at x, or take the better extension
            nMin = min({x, e1, e2});
        }
        curMax = nMax;
        curMin = nMin;
        haveRun = true;
        best = max(best, curMax);          // compare against empty (1) and earlier windows
    }

    cout << best << "\n";
    return 0;
}
