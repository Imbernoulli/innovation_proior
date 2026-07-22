// TIER: strong
// Epoch-graph insight: because K is small, each ship's feasible (entry,exit)
// tick pairs are a tiny enumerable set. Two ships whose chosen tick-index
// ranges never share an epoch need no spatial separation at all, so the
// cheapest way to keep the schedule flexible is to occupy the SHORTEST
// feasible span. Combined with processing highest-value ships first (so a
// rare, space-hungry ship claims its slot before cheap, flexible ships can
// squat on it for the whole day), this reformulates the online packing
// problem into a small per-ship search over epoch choices, with a repair
// pass for anything left over.
#include <bits/stdc++.h>
using namespace std;

int W, H, K;
vector<long long> T;
struct ShipIn { long long r, v, arr, dep; };
vector<ShipIn> S;
struct Acc { int idx; double x, y; int a, b; long long r; };
vector<Acc> accepted;
vector<pair<double, double>> dirs;

bool tryPlace(int idx, long long rr, int a, int b, int maxRings) {
    // small absolute safety margins: the checker tolerates only 1e-6, but
    // our output is rounded to 6 decimals, so never plan a placement that
    // is only exactly (rather than comfortably) feasible.
    double lo_x = rr + 0.02, hi_x = W - rr - 0.02, lo_y = rr + 0.02, hi_y = H - rr - 0.02;
    double cx = W / 2.0, cy = H / 2.0;

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

    if (tryPoint(cx, cy)) return true;
    for (int ring = 1; ring <= maxRings; ring++) {
        double rad = ring * max(1.0, rr * 0.5);
        for (auto &d : dirs) if (tryPoint(cx + d.first * rad, cy + d.second * rad)) return true;
    }
    return false;
}

int main() {
    cin >> W >> H >> K;
    T.assign(K + 1, 0);
    for (int i = 1; i <= K; i++) cin >> T[i];
    int N;
    cin >> N;
    S.assign(N + 1, ShipIn{});
    vector<int> order(N);
    for (int i = 1; i <= N; i++) { cin >> S[i].r >> S[i].v >> S[i].arr >> S[i].dep; order[i - 1] = i; }

    for (int k = 0; k < 8; k++) {
        double ang = 2 * M_PI * k / 8.0;
        dirs.push_back({cos(ang), sin(ang)});
    }

    // highest value first: a scarce, space-hungry ship should claim its slot
    // before flexible, cheap ships can squat on it for the whole horizon.
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (S[a].v != S[b].v) return S[a].v > S[b].v;
        return a < b;
    });

    auto feasibleRange = [&](int idx, int &aMin, int &bMax) {
        aMin = -1; bMax = -1;
        for (int t = 1; t <= K; t++) if (T[t] >= S[idx].arr) { aMin = t; break; }
        for (int t = K; t >= 1; t--) if (T[t] <= S[idx].dep) { bMax = t; break; }
    };

    vector<int> leftover;
    for (int idx : order) {
        long long rr = S[idx].r;
        if (2 * rr > (long long) min(W, H)) continue;
        int aMin, bMax;
        feasibleRange(idx, aMin, bMax);
        if (aMin == -1 || bMax == -1 || aMin >= bMax) { continue; }

        bool placedShip = false;
        // prefer the SHORTEST feasible span (minimises time-overlap with
        // ships placed later), trying every minimal-span entry point.
        for (int a = aMin; a < bMax && !placedShip; a++) {
            int b = a + 1;
            if (tryPlace(idx, rr, a, b, 40)) placedShip = true;
        }
        if (!placedShip && tryPlace(idx, rr, aMin, bMax, 40)) placedShip = true;
        if (!placedShip) leftover.push_back(idx);
    }

    // repair pass: leftover ships get a wider positional search
    for (int idx : leftover) {
        long long rr = S[idx].r;
        if (2 * rr > (long long) min(W, H)) continue;
        int aMin, bMax;
        feasibleRange(idx, aMin, bMax);
        if (aMin == -1 || bMax == -1 || aMin >= bMax) continue;
        bool placedShip = false;
        for (int a = aMin; a < bMax && !placedShip; a++) {
            int b = a + 1;
            if (tryPlace(idx, rr, a, b, 110)) placedShip = true;
        }
        if (!placedShip) tryPlace(idx, rr, aMin, bMax, 110);
    }

    cout << accepted.size() << "\n";
    cout << fixed << setprecision(6);
    for (auto &A : accepted) cout << A.idx << " " << A.x << " " << A.y << " " << A.a << " " << A.b << "\n";
    return 0;
}
