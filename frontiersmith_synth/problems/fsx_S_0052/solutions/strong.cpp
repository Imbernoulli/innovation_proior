// TIER: strong
// Batched nearest-neighbor on a chosen served set, plus a light or-opt style improvement.
// Selects the M cheapest-solo orders (so the served SET is comparable to greedy), then routes
// them with a capacity-aware nearest-neighbor that carries several cuts at once to amortize
// deadhead travel -- reliably beating the one-cut-at-a-time greedy, with per-instance-varying
// behavior driven by the yard's spatial hub structure.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll md(ll ax, ll ay, ll bx, ll by) {
    return llabs(ax - bx) + llabs(ay - by);
}

int main() {
    int N, Q, M;
    scanf("%d %d %d", &N, &Q, &M);
    ll X0, Y0; scanf("%lld %lld", &X0, &Y0);
    vector<ll> ax(N + 1), ay(N + 1), bx(N + 1), by(N + 1), q(N + 1);
    for (int i = 1; i <= N; i++)
        scanf("%lld %lld %lld %lld %lld", &ax[i], &ay[i], &bx[i], &by[i], &q[i]);

    // --- choose served set: M cheapest solo round-trips ---
    vector<pair<ll,int>> solo(N);
    for (int i = 1; i <= N; i++) {
        ll c = md(X0, Y0, ax[i], ay[i]) + md(ax[i], ay[i], bx[i], by[i]) + md(bx[i], by[i], X0, Y0);
        solo[i - 1] = {c, i};
    }
    sort(solo.begin(), solo.end());
    int m = max(1, min(M, N));
    vector<int> served;
    for (int k = 0; k < m; k++) served.push_back(solo[k].second);

    // --- capacity-aware nearest-neighbor over pickups/deliveries ---
    vector<int> pk(N + 1, 0), dl(N + 1, 0);
    ll cx = X0, cy = Y0, load = 0;
    int remaining = 2 * m;
    vector<pair<int,int>> moves;  // (type, idx)
    moves.reserve(2 * m);

    while (remaining > 0) {
        ll best = LLONG_MAX; int bt = -1, bj = -1;
        // deliveries of currently-held cuts (always feasible)
        for (int j : served) {
            if (pk[j] && !dl[j]) {
                ll d = md(cx, cy, bx[j], by[j]);
                if (d < best || (d == best && (bt != 1 || j < bj))) { best = d; bt = 1; bj = j; }
            }
        }
        // pickups of not-yet-collected cuts that fit under the drawbar limit
        for (int j : served) {
            if (!pk[j] && load + q[j] <= Q) {
                ll d = md(cx, cy, ax[j], ay[j]);
                if (d < best || (d == best && !(bt == 1) && j < bj)) { best = d; bt = 0; bj = j; }
            }
        }
        // if nothing fits (loco full and nothing left to pick), force nearest delivery
        if (bt == -1) {
            for (int j : served) {
                if (pk[j] && !dl[j]) {
                    ll d = md(cx, cy, bx[j], by[j]);
                    if (d < best) { best = d; bt = 1; bj = j; }
                }
            }
        }
        if (bt == -1) break;  // safety (should not happen)

        if (bt == 0) { pk[bj] = 1; load += q[bj]; cx = ax[bj]; cy = ay[bj]; }
        else         { dl[bj] = 1; load -= q[bj]; cx = bx[bj]; cy = by[bj]; }
        moves.push_back({bt, bj});
        remaining--;
    }

    // --- light or-opt: relocate a single (pickup,delivery) pair of one order to its best
    //     insertion in the move list if it shortens the tour and keeps feasibility. A few
    //     sweeps; cheap since we only move one order at a time. ---
    auto tourLen = [&](const vector<pair<int,int>>& mv) -> ll {
        ll d = 0, px = X0, py = Y0;
        for (auto& e : mv) {
            ll tx = (e.first == 0) ? ax[e.second] : bx[e.second];
            ll ty = (e.first == 0) ? ay[e.second] : by[e.second];
            d += md(px, py, tx, ty); px = tx; py = ty;
        }
        d += md(px, py, X0, Y0);
        return d;
    };
    // stamped feasibility check: O(len), no per-call allocation
    vector<int> P(N + 1, 0), D(N + 1, 0);
    auto feasible = [&](const vector<pair<int,int>>& mv) -> bool {
        ll ld = 0; bool ok = true;
        for (auto& e : mv) {
            int j = e.second;
            if (e.first == 0) { if (P[j]) ok = false; P[j] = 1; ld += q[j]; if (ld > Q) ok = false; }
            else { if (!P[j] || D[j]) ok = false; D[j] = 1; ld -= q[j]; }
        }
        for (auto& e : mv) { P[e.second] = 0; D[e.second] = 0; }
        if (ld != 0) ok = false;
        return ok;
    };

    int OROPT_CAP = 70;  // exhaustive single-order reinsertion only when the served set is small
    for (int sweep = 0; sweep < 3 && m <= OROPT_CAP; sweep++) {
        bool improved = false;
        for (int j : served) {
            // base = current tour with order j's two moves removed
            vector<pair<int,int>> base;
            base.reserve(moves.size());
            for (auto& e : moves) if (e.second != j) base.push_back(e);
            int S = (int)base.size();
            ll curLen = tourLen(moves);
            ll bestLen = curLen;
            vector<pair<int,int>> bestMv = moves;
            // insert pickup before base index p, delivery before base index d (d >= p)
            for (int p = 0; p <= S; p++) {
                for (int d = p; d <= S; d++) {
                    vector<pair<int,int>> cand;
                    cand.reserve(S + 2);
                    for (int idx = 0; idx < p; idx++) cand.push_back(base[idx]);
                    cand.push_back({0, j});
                    for (int idx = p; idx < d; idx++) cand.push_back(base[idx]);
                    cand.push_back({1, j});
                    for (int idx = d; idx < S; idx++) cand.push_back(base[idx]);
                    if (!feasible(cand)) continue;
                    ll len = tourLen(cand);
                    if (len < bestLen) { bestLen = len; bestMv = cand; }
                }
            }
            if (bestLen < curLen) { moves = bestMv; improved = true; }
        }
        if (!improved) break;
    }

    printf("%d\n", (int)moves.size());
    for (auto& e : moves) printf("%d %d\n", e.first, e.second);
    return 0;
}
