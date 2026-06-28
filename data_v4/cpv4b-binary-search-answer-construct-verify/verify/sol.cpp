#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    // Feasibility: can we place at least k of the sorted positions so that every
    // consecutive chosen pair is at distance >= d? Greedy: always anchor at the
    // smallest position, then take the next position that is >= last_chosen + d.
    // Returns the number of positions placed (capped at k for early exit).
    auto feasible = [&](long long d) -> bool {
        int cnt = 1;                 // position p[0] is always taken first
        long long last = p[0];
        for (int i = 1; i < n; i++) {
            if (p[i] - last >= d) {  // gap large enough: place here
                last = p[i];
                if (++cnt >= k) return true;
            }
        }
        return cnt >= k;
    };

    // Binary search the largest d for which placement of k positions is feasible.
    // d ranges over [1, span]; span = p[n-1] - p[0] is an always-feasible-for-k=2
    // upper-ish bound, but the true max min-gap never exceeds span/(k-1), so cap there.
    long long lo = 1, hi = (p[n - 1] - p[0]) / (k - 1);
    if (hi < 1) hi = 1;
    long long best = 1;              // d = 1 is feasible whenever k <= n (distinct positions)
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
    }

    // Reconstruct one witness placement realizing the optimal d = best.
    vector<long long> chosen;
    chosen.push_back(p[0]);
    long long last = p[0];
    for (int i = 1; i < n && (int)chosen.size() < k; i++) {
        if (p[i] - last >= best) {
            chosen.push_back(p[i]);
            last = p[i];
        }
    }

    cout << best << "\n";
    for (int i = 0; i < (int)chosen.size(); i++) {
        cout << chosen[i] << (i + 1 < (int)chosen.size() ? ' ' : '\n');
    }
    return 0;
}
