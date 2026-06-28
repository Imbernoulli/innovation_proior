#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k;
    if (!(cin >> n >> k)) return 0;

    const long long INF = (long long)4e18; // "impossible" sentinel, far above any real cost

    if (n == 0) { cout << 0 << "\n"; return 0; } // no houses -> nothing to pay

    // prev[c] = min cost to paint houses 0..i with house i painted color c.
    // The first house has no predecessor, so its cost is just its own paint cost.
    vector<long long> prev(k);
    for (long long c = 0; c < k; c++) cin >> prev[c];

    for (long long i = 1; i < n; i++) {
        // Two smallest values of prev[], with the index of the smallest.
        // best1 = smallest prev value, idx1 its color; best2 = second smallest (idx1 excluded).
        long long best1 = INF, best2 = INF, idx1 = -1;
        for (long long c = 0; c < k; c++) {
            if (prev[c] < best1) { best2 = best1; best1 = prev[c]; idx1 = c; }
            else if (prev[c] < best2) { best2 = prev[c]; }
        }

        vector<long long> cur(k);
        for (long long c = 0; c < k; c++) {
            long long cost;
            cin >> cost;
            // cheapest previous-house entry painted a DIFFERENT color than c
            long long bestPrevOther = (c == idx1) ? best2 : best1;
            if (bestPrevOther >= INF) cur[c] = INF;          // no legal predecessor (e.g. k == 1)
            else cur[c] = bestPrevOther + cost;
        }
        prev = move(cur);
    }

    long long ans = INF;
    for (long long c = 0; c < k; c++) ans = min(ans, prev[c]);
    cout << (ans >= INF ? -1 : ans) << "\n";
    return 0;
}
