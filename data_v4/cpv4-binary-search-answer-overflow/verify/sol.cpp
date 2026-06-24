#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    long long N;
    if (!(cin >> m >> N)) return 0;
    vector<long long> w(m), c(m);
    for (int i = 0; i < m; i++) cin >> w[i] >> c[i];

    if (N == 0) { cout << 0 << "\n"; return 0; }

    // produced(T): total parts all presses make by time T (milliseconds).
    // press i: 0 if T < w[i], else floor((T - w[i]) / c[i]) + 1.
    // Returns min(total, CAP) so the running sum cannot overflow long long.
    const long long CAP = (long long)4e18;
    auto produced = [&](long long T) -> long long {
        long long total = 0;
        for (int i = 0; i < m; i++) {
            if (T >= w[i]) {
                total += (T - w[i]) / c[i] + 1;
                if (total >= CAP) return CAP; // saturate early
            }
        }
        return total;
    };

    // Binary search smallest T with produced(T) >= N.
    // hi: a T that is certainly enough. The fastest press finishes N parts at
    // w_min + (N-1)*c_min; bound generously with max possible values.
    long long lo = 0;
    long long hi = 0;
    for (int i = 0; i < m; i++) {
        long long t = w[i] + (N - 1) * c[i]; // press i alone makes N parts by here
        hi = max(hi, t);
    }
    // hi as computed is an upper bound (single press alone already reaches N).

    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (produced(mid) >= N) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
