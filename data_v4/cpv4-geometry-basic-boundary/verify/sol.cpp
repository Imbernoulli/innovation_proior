#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<long long> X1(n), Y1(n), X2(n), Y2(n);
    // Collect column boundaries. A rectangle covers integer columns x1..x2 inclusive.
    // We sweep over unit-column strips. To use coordinate compression on a closed
    // (inclusive) integer grid, expand each rectangle's x-range to the half-open
    // interval [x1, x2+1): integer columns x1..x2 correspond to unit strips
    // [x1,x1+1), ..., [x2,x2+1). The same trick is applied to y inside each strip.
    vector<long long> xs;
    xs.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        long long a, b, c, d;
        cin >> a >> b >> c >> d; // (a,b)-(c,d) opposite corners, any order
        long long x1 = min(a, c), x2 = max(a, c);
        long long y1 = min(b, d), y2 = max(b, d);
        X1[i] = x1; Y1[i] = y1; X2[i] = x2; Y2[i] = y2;
        xs.push_back(x1);
        xs.push_back(x2 + 1); // half-open right end
    }
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());

    long long total = 0;
    // Sweep each x-slab [xs[k], xs[k+1]) which is a band of (xs[k+1]-xs[k]) integer columns.
    for (size_t k = 0; k + 1 < xs.size(); k++) {
        long long xl = xs[k];
        long long xr = xs[k + 1];
        long long width = xr - xl; // number of integer columns in this slab
        if (width <= 0) continue;

        // Gather y-intervals of rectangles whose x-range covers this slab.
        // Rectangle i covers columns [X1[i], X2[i]]; in half-open terms [X1[i], X2[i]+1).
        // It covers the slab iff X1[i] <= xl and X2[i]+1 >= xr.
        vector<pair<long long,long long>> ivs; // half-open [y1, y2+1)
        for (int i = 0; i < n; i++) {
            if (X1[i] <= xl && X2[i] + 1 >= xr) {
                ivs.push_back({Y1[i], Y2[i] + 1});
            }
        }
        if (ivs.empty()) continue;
        sort(ivs.begin(), ivs.end());

        // Union length of half-open y-intervals = number of distinct integer rows covered.
        long long curL = ivs[0].first, curR = ivs[0].second;
        long long rows = 0;
        for (size_t j = 1; j < ivs.size(); j++) {
            if (ivs[j].first > curR) {
                rows += curR - curL;
                curL = ivs[j].first;
                curR = ivs[j].second;
            } else {
                curR = max(curR, ivs[j].second);
            }
        }
        rows += curR - curL;

        total += width * rows;
    }

    cout << total << "\n";
    return 0;
}
