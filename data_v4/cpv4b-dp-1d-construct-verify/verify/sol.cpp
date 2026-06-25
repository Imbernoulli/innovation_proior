#include <bits/stdc++.h>
using namespace std;

// canFill(m, hh, c): can we fill m remaining slots with exactly hh hits (and m-hh rests),
// no run of K consecutive rests, given the run of rests ALREADY standing immediately
// before slot 0 has length c (0 <= c <= K-1)?
//
// hh hits split the m-hh rests into hh+1 gaps g_0,...,g_hh. The first gap g_0 is glued to
// the existing run c, so g_0 + c <= K-1; every other gap g_i <= K-1. Hence the maximum
// number of rests that fit is (K-1-c) + hh*(K-1) = (hh+1)*(K-1) - c. So feasibility is:
//   0 <= hh <= m  AND  c <= K-1  AND  (m - hh) <= (long long)(hh+1)*(K-1) - c
// Everything is 64-bit: (hh+1)*(K-1) can reach ~1e14.
static inline bool canFill(long long m, long long hh, long long c, long long K) {
    if (hh < 0 || hh > m) return false;
    if (c > K - 1) return false;
    long long capacity = (hh + 1) * (K - 1) - c; // max rests that fit
    return (m - hh) <= capacity;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, K, h;
    if (!(cin >> n >> K >> h)) return 0;

    // Global feasibility: fill all n slots with h hits, starting with no standing rest run.
    if (!canFill(n, h, 0, K)) {
        cout << "-1\n";
        return 0;
    }

    // Greedy: build lexicographically smallest pattern with 'H' < 'R'. Prefer 'H'; place it
    // whenever a hit is left and the remaining suffix stays completable. Otherwise place 'R'
    // (allowed only if it does not reach a run of K and the suffix stays completable).
    string out;
    out.reserve((size_t)n);
    long long hitsLeft = h;
    long long c = 0; // current trailing run of rests
    for (long long pos = 0; pos < n; pos++) {
        long long m = n - pos - 1; // slots remaining AFTER this one
        if (hitsLeft >= 1 && canFill(m, hitsLeft - 1, 0, K)) {
            out.push_back('H');
            hitsLeft -= 1;
            c = 0;
        } else {
            // place 'R'; invariant guarantees this branch is feasible
            out.push_back('R');
            c += 1;
        }
    }

    out.push_back('\n');
    cout << out;
    return 0;
}
