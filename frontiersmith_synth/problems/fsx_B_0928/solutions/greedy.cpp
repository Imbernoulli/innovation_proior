// TIER: greedy
// The "obvious" approach: process ships in arrival order, read each ship's
// window literally (enter as soon as allowed, leave as late as allowed --
// the maximal span), and place its anchor at the basin centre if free, else
// spiral outward. No look-ahead, no reuse planning.
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
    vector<int> order(N);
    for (int i = 1; i <= N; i++) { cin >> r[i] >> v[i] >> arr[i] >> dep[i]; order[i - 1] = i; }

    sort(order.begin(), order.end(), [&](int a, int b) {
        if (arr[a] != arr[b]) return arr[a] < arr[b];
        return a < b;
    });

    vector<pair<double, double>> dirs;
    for (int k = 0; k < 8; k++) {
        double ang = 2 * M_PI * k / 8.0;
        dirs.push_back({cos(ang), sin(ang)});
    }

    struct Acc { int idx; double x, y; int a, b; long long r; };
    vector<Acc> accepted;
    double cx = W / 2.0, cy = H / 2.0;

    for (int idx : order) {
        long long rr = r[idx];
        if (2 * rr > (long long) min(W, H)) continue;
        int a = -1, b = -1;
        for (int t = 1; t <= K; t++) if (T[t] >= arr[idx]) { a = t; break; }
        for (int t = K; t >= 1; t--) if (T[t] <= dep[idx]) { b = t; break; }
        if (a == -1 || b == -1 || a >= b) continue;

        // small absolute safety margins: the checker tolerates only 1e-6, but
        // our output is rounded to 6 decimals, so never plan a placement
        // that is only exactly (rather than comfortably) feasible.
        double lo_x = rr + 0.02, hi_x = W - rr - 0.02, lo_y = rr + 0.02, hi_y = H - rr - 0.02;
        bool placed = false;

        auto tryPoint = [&](double px, double py) -> bool {
            if (px < lo_x || px > hi_x || py < lo_y || py > hi_y) return false;
            for (auto &A : accepted) {
                if (max(a, A.a) < min(b, A.b)) {
                    double dx = px - A.x, dy = py - A.y;
                    double need = (double) (rr + A.r) + 0.05;
                    if (dx * dx + dy * dy < need * need) return false;
                }
            }
            accepted.push_back({idx, px, py, a, b, rr});
            return true;
        };

        if (tryPoint(cx, cy)) { placed = true; }
        for (int ring = 1; ring <= 60 && !placed; ring++) {
            double rad = ring * max(1.0, rr * 0.5);
            for (auto &d : dirs) {
                if (tryPoint(cx + d.first * rad, cy + d.second * rad)) { placed = true; break; }
            }
        }
        // ship rejected silently if never placed
    }

    cout << accepted.size() << "\n";
    cout << fixed << setprecision(6);
    for (auto &A : accepted) cout << A.idx << " " << A.x << " " << A.y << " " << A.a << " " << A.b << "\n";
    return 0;
}
