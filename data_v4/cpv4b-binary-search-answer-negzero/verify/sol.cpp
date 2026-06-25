#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> x(n);
    for (auto &v : x) cin >> v;

    sort(x.begin(), x.end());

    // feasible(d): can we install k sensors so every pair is at least d apart?
    // Greedy on the sorted coordinates: always keep the first borehole, then take the
    // next borehole whose coordinate is at least (last placed + d).
    auto feasible = [&](long long d) -> bool {
        int cnt = 1;                 // place the first (leftmost) sensor
        long long last = x[0];
        for (int i = 1; i < n; i++) {
            if (x[i] - last >= d) {
                cnt++;
                last = x[i];
            }
        }
        return cnt >= k;
    };

    // The optimal isolation is some adjacent-difference value, in [0, span].
    long long lo = 0, hi = x[n - 1] - x[0];

    long long ans = 0;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }

    cout << ans << "\n";
    return 0;
}
