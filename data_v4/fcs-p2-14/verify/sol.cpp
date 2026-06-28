#include <bits/stdc++.h>
using namespace std;

// Maximum sum of a no-two-adjacent subset on a LINE over a[lo..hi] (inclusive).
// Empty subset allowed, so the returned value is always >= 0.
// If lo > hi the range is empty and the best sum is 0.
static long long linearBest(const vector<long long> &a, int lo, int hi) {
    long long take = LLONG_MIN / 4; // best with last position taken (impossible before any element)
    long long skip = 0;             // best with last position not taken (empty prefix -> 0)
    for (int i = lo; i <= hi; i++) {
        long long ntake = skip + a[i];     // take i => i-1 skipped
        long long nskip = max(skip, take); // skip i => i-1 either
        take = ntake;
        skip = nskip;
    }
    return max({take, skip, 0LL});
}

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }
    if (n == 1) { cout << max(a[0], 0LL) << "\n"; return 0; }

    // Circle: positions 0 and n-1 are adjacent, so they cannot both be chosen.
    // Split into two LINE subproblems that each break the wrap edge:
    //   (A) forbid the last element  -> solve line over a[0 .. n-2]
    //   (B) forbid the first element -> solve line over a[1 .. n-1]
    // Any valid circular selection avoids at least one of {first, last}, so it is
    // covered by case A or case B; conversely any selection counted in A or B is a
    // valid line selection that never uses both endpoints, hence valid on the circle.
    long long best = max(linearBest(a, 0, n - 2), linearBest(a, 1, n - 1));

    cout << best << "\n"; // empty selection always allowed (linearBest already >= 0)
    return 0;
}
