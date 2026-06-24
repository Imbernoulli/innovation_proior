#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L, D;
    if (!(cin >> n >> L >> D)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    long long total = (long long)n * (n - 1) / 2;

    // If 2D >= L every unordered pair is within circular distance D.
    if (2 * D >= L) {
        cout << total << "\n";
        return 0;
    }

    // 2D < L: the "close" condition (d <= D) and the "far" condition
    // (d >= L-D) are disjoint, so count each with a separate sweep and add.
    long long ans = 0;

    // Close: number of pairs (i<j) with p[j]-p[i] <= D.
    {
        int lo = 0;
        for (int hi = 0; hi < n; hi++) {
            while (p[hi] - p[lo] > D) lo++;
            ans += (long long)(hi - lo); // pairs (lo..hi-1, hi)
        }
    }

    // Far: number of pairs (i<j) with p[j]-p[i] >= L-D.
    {
        long long thr = L - D;
        int lo = 0;
        for (int hi = 0; hi < n; hi++) {
            while (p[hi] - p[lo] >= thr) lo++;
            // indices [0 .. lo-1] satisfy p[hi]-p[i] >= thr
            ans += (long long)lo;
        }
    }

    cout << ans << "\n";
    return 0;
}
