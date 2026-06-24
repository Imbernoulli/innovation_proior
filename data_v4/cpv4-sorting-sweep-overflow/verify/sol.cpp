#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    // Each stream i is active on the half-open interval [s_i, e_i) with weight w_i.
    // Build sweep events: at time s a +w event, at time e a -w event.
    // Sort by time; at equal time, process the -w (end) events before the +w (start)
    // events so that a stream ending exactly when another starts does NOT overlap it.
    vector<pair<long long, long long>> ev;
    ev.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        long long s, e, w;
        cin >> s >> e >> w;
        if (s >= e) continue;              // empty interval contributes nothing
        ev.push_back({s, +w});             // start: add weight
        ev.push_back({e, -w});             // end:   remove weight
    }

    // Sort by time. For equal time, ends (negative delta) come before starts (positive
    // delta). Sorting the pair (time, delta) ascending does exactly that, because a
    // negative delta sorts before a positive one at the same time.
    sort(ev.begin(), ev.end());

    long long cur = 0, best = 0;           // 64-bit: load can reach ~2e5 * 1e9 = 2e14
    for (auto &p : ev) {
        cur += p.second;
        best = max(best, cur);
    }

    cout << best << "\n";
    return 0;
}
