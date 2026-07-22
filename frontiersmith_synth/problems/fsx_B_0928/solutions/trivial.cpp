// TIER: trivial
// Accepts exactly the single best feasible ship (the same construction the
// checker uses as its internal baseline B) and rejects everyone else.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int W, H, K;
    cin >> W >> H >> K;
    vector<long long> T(K + 1);
    for (int i = 1; i <= K; i++) cin >> T[i];
    int N;
    cin >> N;
    vector<long long> r(N + 1), v(N + 1), arr(N + 1), dep(N + 1);
    for (int i = 1; i <= N; i++) cin >> r[i] >> v[i] >> arr[i] >> dep[i];

    int best = -1;
    long long bestV = -1;
    int bestA = -1, bestB = -1;
    for (int i = 1; i <= N; i++) {
        if (2 * r[i] > (long long) min(W, H)) continue;
        int a = -1, b = -1;
        for (int t = 1; t <= K; t++) if (T[t] >= arr[i]) { a = t; break; }
        for (int t = K; t >= 1; t--) if (T[t] <= dep[i]) { b = t; break; }
        if (a == -1 || b == -1 || a >= b) continue;
        if (v[i] > bestV) { bestV = v[i]; best = i; bestA = a; bestB = b; }
    }

    if (best == -1) {
        cout << 0 << "\n";
        return 0;
    }
    double cx = W / 2.0, cy = H / 2.0;
    cout << 1 << "\n";
    cout << best << " " << fixed << setprecision(6) << cx << " " << cy << " " << bestA << " " << bestB << "\n";
    return 0;
}
