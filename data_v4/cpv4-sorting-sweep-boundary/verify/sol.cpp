#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 -> no cars -> answer 0

    // Each car occupies the half-open interval [s, e): present at s, gone by e.
    // A car ending at time t and a car starting at time t do NOT overlap.
    // We build events: a start contributes +1 AT coordinate s; an end contributes
    // -1 AT coordinate e. We sweep coordinates in increasing order. At a shared
    // coordinate, ENDS must be applied before STARTS, because [a,t) and [t,b) are
    // disjoint -- the leaving car frees the instant before the arriving car claims it.
    // event = (coordinate, type) with type 0 = end (-1), type 1 = start (+1).
    vector<pair<long long,int>> ev;
    ev.reserve((size_t)2 * n);
    for (int i = 0; i < n; i++) {
        long long s, e;
        cin >> s >> e;
        // Guard: a degenerate interval with s >= e occupies no instant; skip it.
        if (s >= e) continue;
        ev.push_back({s, 1});   // start
        ev.push_back({e, 0});   // end
    }

    // Sort by coordinate ascending; within equal coordinate, type 0 (end) before
    // type 1 (start). Since we encode end as 0 and start as 1, a plain pair sort
    // on (coordinate, type) gives ends first at ties -- exactly what half-open needs.
    sort(ev.begin(), ev.end());

    long long cur = 0, best = 0;
    for (auto &p : ev) {
        if (p.second == 1) cur += 1; else cur -= 1;
        best = max(best, cur);
    }

    cout << best << "\n";
    return 0;
}
