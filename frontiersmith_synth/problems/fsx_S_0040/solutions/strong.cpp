// TIER: strong
// Cheapest-insertion construction + removal/re-insertion local search with
// randomized multi-restart. Each restart greedily inserts the most profitable
// job at its best (pickup gap, delivery gap) positions, then improves by
// dropping unprofitable served jobs and re-inserting; the lowest-cost feasible
// tour over all restarts is emitted.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, depot;
vector<int> X, Y, D, P, Q, W;
ll B = 0;

static inline ll darr(int a, int b) {
    return (ll)abs(X[a] - X[b]) + abs(Y[a] - Y[b]) + D[b];
}

// events: (type, job). locations derived from events.
static void buildLocs(const vector<pair<int,int>>& ev, vector<int>& locs) {
    locs.clear();
    locs.reserve(ev.size());
    for (auto& e : ev) locs.push_back(e.first == 1 ? P[e.second] : Q[e.second]);
}

static ll routeCost(const vector<int>& locs) {
    if (locs.empty()) return 0;
    ll c = darr(depot, locs.front());
    for (size_t i = 1; i < locs.size(); i++) c += darr(locs[i - 1], locs[i]);
    c += darr(locs.back(), depot);
    return c;
}

static ll objOf(const vector<pair<int,int>>& ev) {
    vector<int> locs; buildLocs(ev, locs);
    ll served = 0;
    vector<char> srv(M + 1, 0);
    for (auto& e : ev) if (e.first == 1) { srv[e.second] = 1; }
    for (int j = 1; j <= M; j++) if (srv[j]) served += W[j];
    return routeCost(locs) + (B - served);
}

mt19937 rng(20260701u);

int main() {
    if (scanf("%d %d %d", &N, &M, &depot) != 3) return 0;
    X.assign(N + 1, 0); Y.assign(N + 1, 0); D.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%d %d %d", &X[i], &Y[i], &D[i]);
    P.assign(M + 1, 0); Q.assign(M + 1, 0); W.assign(M + 1, 0);
    for (int j = 1; j <= M; j++) { scanf("%d %d %d", &P[j], &Q[j], &W[j]); B += W[j]; }

    vector<pair<int,int>> best;         // best events overall
    ll bestF = B;                       // do-nothing baseline

    auto seqLocAt = [&](const vector<int>& locs, int g) -> int {
        int len = (int)locs.size();
        if (g <= 0) return depot;
        if (g >= len + 1) return depot;
        return locs[g - 1];
    };

    // best insertion of job j into current events; returns delta and positions.
    auto bestInsertion = [&](const vector<int>& locs, int j, ll& outDelta,
                             int& gi, int& gj) {
        int len = (int)locs.size();
        outDelta = LLONG_MAX; gi = gj = 0;
        for (int i = 0; i <= len; i++) {
            int a = seqLocAt(locs, i);
            int an = seqLocAt(locs, i + 1);
            for (int b = i; b <= len; b++) {
                ll delta;
                if (b == i) {
                    delta = darr(a, P[j]) + darr(P[j], Q[j]) + darr(Q[j], an) - darr(a, an);
                } else {
                    int c = seqLocAt(locs, b);
                    int cn = seqLocAt(locs, b + 1);
                    delta = (darr(a, P[j]) + darr(P[j], an) - darr(a, an))
                          + (darr(c, Q[j]) + darr(Q[j], cn) - darr(c, cn));
                }
                if (delta < outDelta) { outDelta = delta; gi = i; gj = b; }
            }
        }
    };

    auto doInsert = [&](vector<pair<int,int>>& ev, int j, int gi, int gj) {
        ev.insert(ev.begin() + gi, {1, j});
        ev.insert(ev.begin() + (gj + 1), {2, j});
    };

    int restarts = 12;
    for (int rs = 0; rs < restarts; rs++) {
        vector<pair<int,int>> ev;
        vector<char> served(M + 1, 0);

        // ---- greedy cheapest-insertion construction ----
        while (true) {
            vector<int> locs; buildLocs(ev, locs);
            // gather candidate savings for all unserved jobs
            ll bestSave = 0; int bestJ = -1, bi = 0, bj = 0;
            vector<tuple<ll,int,int,int>> pos; // (save, j, gi, gj) for exploration
            for (int j = 1; j <= M; j++) {
                if (served[j]) continue;
                ll delta; int gi, gj;
                bestInsertion(locs, j, delta, gi, gj);
                ll save = W[j] - delta;
                if (save > 0) pos.push_back({save, j, gi, gj});
                if (save > bestSave) { bestSave = save; bestJ = j; bi = gi; bj = gj; }
            }
            if (bestJ == -1) break; // no profitable insertion left
            int pj = bestJ, pi = bi, pjj = bj;
            // occasional exploration among positive-savings candidates
            if (rs > 0 && !pos.empty() && (rng() % 100) < 25) {
                auto& t = pos[rng() % pos.size()];
                pj = get<1>(t); pi = get<2>(t); pjj = get<3>(t);
            }
            doInsert(ev, pj, pi, pjj);
            served[pj] = 1;
        }

        // ---- local search: drop unprofitable served jobs, then re-insert ----
        for (int pass = 0; pass < 3; pass++) {
            bool changed = false;
            // removal pass
            for (int j = 1; j <= M; j++) {
                if (!served[j]) continue;
                vector<pair<int,int>> cand;
                for (auto& e : ev) if (e.second != j) cand.push_back(e);
                if (objOf(cand) < objOf(ev)) { ev = cand; served[j] = 0; changed = true; }
            }
            // re-insertion pass
            while (true) {
                vector<int> locs; buildLocs(ev, locs);
                ll bestSave = 0; int bestJ = -1, bi = 0, bj = 0;
                for (int j = 1; j <= M; j++) {
                    if (served[j]) continue;
                    ll delta; int gi, gj;
                    bestInsertion(locs, j, delta, gi, gj);
                    ll save = W[j] - delta;
                    if (save > bestSave) { bestSave = save; bestJ = j; bi = gi; bj = gj; }
                }
                if (bestJ == -1) break;
                doInsert(ev, bestJ, bi, bj);
                served[bestJ] = 1; changed = true;
            }
            if (!changed) break;
        }

        ll F = objOf(ev);
        if (F < bestF) { bestF = F; best = ev; }
    }

    printf("%d\n", (int)best.size());
    for (auto& e : best) printf("%d %d\n", e.first, e.second);
    return 0;
}
