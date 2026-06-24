#include <bits/stdc++.h>
using namespace std;

// Disjoint-set "latest free day": find(x) returns the largest day index <= x
// that is still free (0 means no free day at or before x).
static vector<int> par;
int findFree(int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;                 // n = 0 (or empty input) -> answer 0
    vector<long long> d(n), v(n);
    for (int i = 0; i < n; i++) cin >> d[i];
    for (int i = 0; i < n; i++) cin >> v[i];

    // Only positive-payout gigs are ever worth scheduling; a gig with v <= 0
    // can always be declined for a strictly-not-worse total, so we skip it.
    // Among positive gigs, sort by payout descending and assign each to the
    // latest still-free day at or before its deadline (greedy exchange).
    vector<int> order;
    order.reserve(n);
    long long maxDay = 0;
    for (int i = 0; i < n; i++) {
        if (v[i] > 0) {
            order.push_back(i);
            // a deadline beyond n is useless: at most n gigs fit, so cap at n.
            long long cap = min(d[i], (long long)n);
            if (cap > maxDay) maxDay = cap;
        }
    }
    sort(order.begin(), order.end(), [&](int a, int b) { return v[a] > v[b]; });

    par.assign((size_t)maxDay + 1, 0);
    for (int day = 0; day <= (int)maxDay; day++) par[day] = day;

    long long answer = 0;
    for (int idx : order) {
        long long cap = min(d[idx], maxDay);   // latest day this gig may occupy
        if (cap <= 0) continue;                // deadline 0 -> no valid day
        int slot = findFree((int)cap);
        if (slot > 0) {                        // a free day exists
            answer += v[idx];
            par[slot] = slot - 1;              // mark day `slot` used
        }
    }

    cout << answer << "\n";                     // empty / all-nonpositive -> 0
    return 0;
}
