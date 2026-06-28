#include <bits/stdc++.h>
using namespace std;

// Print a (possibly large) non-negative 128-bit integer.
static void print128(__int128 x) {
    if (x == 0) { cout << "0\n"; return; }
    string s;
    while (x > 0) { s += char('0' + (int)(x % 10)); x /= 10; }
    reverse(s.begin(), s.end());
    cout << s << "\n";
}

// Minimum total weighted Manhattan distance from a chosen integer meeting point
// to all weighted sites. L1 separates across axes, so we solve each axis with a
// weighted median: sort the (coordinate, weight) pairs, walk the prefix of weight
// until it reaches half the total, and that coordinate is an optimal meeting value.
// The total cost can reach ~4e23, so accumulate the answer in __int128.
static __int128 solveAxis(vector<pair<long long,long long>> &v, long long total) {
    sort(v.begin(), v.end());
    // Weighted median coordinate m: smallest coord whose prefix weight reaches
    // ceil(total/2). Any optimum can be taken at such a site coordinate.
    long long need = (total + 1) / 2;
    long long acc = 0;
    long long m = v.empty() ? 0 : v.back().first;
    for (auto &p : v) {
        acc += p.second;
        if (acc >= need) { m = p.first; break; }
    }
    __int128 cost = 0;
    for (auto &p : v) {
        long long d = llabs(p.first - m);
        cost += (__int128)d * (__int128)p.second;
    }
    return cost;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<pair<long long,long long>> xs(n), ys(n);
    long long total = 0;
    for (int i = 0; i < n; i++) {
        long long x, y, w;
        cin >> x >> y >> w;
        xs[i] = {x, w};
        ys[i] = {y, w};
        total += w;
    }

    if (n == 0 || total == 0) {
        cout << 0 << "\n";
        return 0;
    }

    __int128 ans = solveAxis(xs, total) + solveAxis(ys, total);
    print128(ans);
    return 0;
}
