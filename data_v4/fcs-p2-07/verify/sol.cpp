#include <bits/stdc++.h>
using namespace std;

int main() {
    int k;       // number of eggs, 1 <= k <= 100
    long long m; // number of floors, 1 <= m <= 10000
    if (!(cin >> k >> m)) return 0;

    // cover[e] = maximum number of floors distinguishable with e eggs and the
    // current number of trials t. Recurrence over one more trial:
    //   cover_t(e) = cover_{t-1}(e-1) + cover_{t-1}(e) + 1
    // (drop from a floor: if the egg breaks we have e-1 eggs and t-1 trials for
    //  the floors below; if it survives we have e eggs and t-1 trials for the
    //  floors above; plus the floor we dropped from). cover_0(e) = 0 for all e,
    //  and cover_t(0) = 0 for all t. We want the smallest t with cover_t(k) >= m.
    //
    // We cap the working egg count at min(k, 14): with t trials the most floors
    // any number of eggs can cover is 2^t - 1, and 2^14 - 1 = 16383 >= 10000 = m,
    // so the answer never exceeds 14 once eggs >= 14. With at most 14 trials an
    // egg beyond the 14th can never be dropped, so eggs past 14 are irrelevant
    // and capping the working egg count at 14 leaves the answer unchanged.
    int kk = (int)min<long long>(k, 14);
    vector<long long> cover(kk + 1, 0);
    long long t = 0;
    while (cover[kk] < m) {
        ++t;
        // update from high egg count to low so cover[e-1] still holds the
        // previous-trial value when we read it.
        for (int e = kk; e >= 1; --e) {
            long long below = cover[e - 1];           // breaks: e-1 eggs, t-1 trials
            long long above = cover[e];               // survives: e eggs, t-1 trials
            long long c = below + above + 1;
            if (c > m) c = m;                          // clamp to avoid overflow
            cover[e] = c;
        }
    }

    cout << t << "\n";
    return 0;
}
