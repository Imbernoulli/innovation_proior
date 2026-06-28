#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<pair<long long, long long>> p(n); // (x, y)
    for (auto &q : p) cin >> q.first >> q.second;

    if (n < 2) {                       // fewer than two points: no pair exists
        cout << -1 << "\n";
        return 0;
    }

    sort(p.begin(), p.end());          // sort by x, then y

    // best = current smallest squared distance found so far.
    long long best = LLONG_MAX;

    // Active set: points within the vertical strip of half-width d = sqrt(best)
    // behind the sweep, keyed by (y, x) so we can range-query on y.
    set<pair<long long, long long>> win; // (y, x)
    int left = 0;                      // p[left..i-1] are candidates by x
    for (int i = 0; i < n; i++) {
        long long xi = p[i].first, yi = p[i].second;

        // d = ceil(sqrt(best)); any point farther than d in x cannot beat best.
        // Use the squared comparison directly: drop p[left] while
        // (xi - p[left].x)^2 >= best  (strictly cannot improve).
        while (left < i) {
            long long dx = xi - p[left].first;
            if (dx * dx >= best) {
                win.erase({p[left].second, p[left].first});
                left++;
            } else break;
        }

        // Window in y: [yi - d, yi + d]. We don't have d as an integer, so we
        // iterate the set on the y-band and stop once dy^2 >= best.
        // Lower bound: first point with y >= yi - d. Since d^2 ~ best, use
        // a sentinel; we walk outward from yi using lower_bound on y.
        if (best == LLONG_MAX) {
            // No bound yet: just scan everything currently in window
            // (only happens for the very first comparisons; window is tiny).
            for (auto &q : win) {
                long long dx = xi - q.second;
                long long dy = yi - q.first;
                long long d2 = dx * dx + dy * dy;
                if (d2 < best) best = d2;
            }
        } else {
            // Integer ceil of sqrt(best) gives the y-band half-width.
            long long d = (long long)ceil(sqrt((double)best));
            // guard against floating error
            while (d * d < best) d++;
            while (d > 0 && (d - 1) * (d - 1) >= best) d--;

            auto lo = win.lower_bound({yi - d, LLONG_MIN});
            auto hi = win.upper_bound({yi + d, LLONG_MAX});
            for (auto it = lo; it != hi; ++it) {
                long long dx = xi - it->second;
                long long dy = yi - it->first;
                long long d2 = dx * dx + dy * dy;
                if (d2 < best) best = d2;
            }
        }

        win.insert({yi, xi});
    }

    cout << best << "\n";
    return 0;
}
