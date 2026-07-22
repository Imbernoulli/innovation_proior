// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// The insight this problem is built around: the unit of decision is the
// RENDEZVOUS, not the per-scope move. Reformulate as a time-expanded
// pair-assignment graph -- repeatedly evaluate every (target, site-0 scope,
// site-1 scope) triple by the payout it would achieve from BOTH scopes'
// CURRENT tail state, commit the single best triple (advancing both
// scopes' timelines), and repeat. This directly targets what actually pays
// out (a completed cross-site pair) instead of per-scope apparent value,
// and naturally sequences each scope's slew route as a side effect of the
// commitment order.

static int H;
static ll Pn, Qd;

static inline int angdist(int a, int b) { int d = abs(a - b) % 360; return min(d, 360 - d); }
static inline int travelTicks(int p, int q, int speed) {
    int d = angdist(p, q);
    if (d == 0) return 0;
    return (d + speed - 1) / speed;
}
static inline ll decayVal(ll v, int dt) {
    for (int d = 0; d < dt; d++) { v = (v * Pn) / Qd; if (v <= 0) return 0; }
    return v;
}

int main() {
    int T, M, W;
    cin >> T >> M >> H >> W >> Pn >> Qd;
    vector<int> site(T), pos(T), speed(T);
    for (int i = 0; i < T; i++) cin >> site[i] >> pos[i] >> speed[i];
    vector<int> a(M), pt(M), v(M), o(M);
    for (int j = 0; j < M; j++) cin >> a[j] >> pt[j] >> v[j] >> o[j];

    vector<int> s0, s1;
    for (int i = 0; i < T; i++) (site[i] == 0 ? s0 : s1).push_back(i);

    vector<int> curT(T, 0), curPos(T);
    for (int i = 0; i < T; i++) curPos[i] = pos[i];
    vector<vector<pair<int,int>>> visits(T);
    vector<char> used(M, 0);

    int rounds = M + 2;
    for (int r = 0; r < rounds; r++) {
        ll bestVal = -1;
        int bestJ = -1, bestU = -1, bestV = -1, bestArrU = -1, bestArrV = -1;
        for (int j = 0; j < M; j++) {
            if (used[j]) continue;
            for (int u : s0) {
                int arrU = curT[u] + travelTicks(curPos[u], pt[j], speed[u]);
                if (arrU + o[j] > H || arrU < a[j]) continue;
                for (int w : s1) {
                    int arrV = curT[w] + travelTicks(curPos[w], pt[j], speed[w]);
                    if (arrV + o[j] > H || arrV < a[j]) continue;
                    if (abs(arrU - arrV) > W) continue;
                    int pairTick = max(arrU, arrV);
                    ll val = decayVal(v[j], pairTick - a[j]);
                    if (val > bestVal) {
                        bestVal = val; bestJ = j; bestU = u; bestV = w;
                        bestArrU = arrU; bestArrV = arrV;
                    }
                }
            }
        }
        if (bestJ < 0 || bestVal <= 0) break;
        visits[bestU].push_back({bestArrU, bestJ});
        curT[bestU] = bestArrU + o[bestJ]; curPos[bestU] = pt[bestJ];
        visits[bestV].push_back({bestArrV, bestJ});
        curT[bestV] = bestArrV + o[bestJ]; curPos[bestV] = pt[bestJ];
        used[bestJ] = 1;
    }

    // Idle scopes (or ones done pairing) mop up their own zero-travel
    // personal lure if still available and unclaimed by anyone -- harmless
    // extra output, never hurts (never pairs, but never subtracts either).
    for (int i = 0; i < T; i++) {
        int bestJ = -1, bestArr = -1; ll bestVal = -1;
        int cap = 20 - (int)visits[i].size();
        for (int iter = 0; iter < max(0, cap); iter++) {
            bestJ = -1; bestVal = -1;
            for (int j = 0; j < M; j++) {
                if (used[j]) continue;
                int travel = travelTicks(curPos[i], pt[j], speed[i]);
                int arr = curT[i] + travel;
                if (arr + o[j] > H || arr < a[j]) continue;
                ll val = decayVal(v[j], arr - a[j]);
                if (val > bestVal) { bestVal = val; bestJ = j; bestArr = arr; }
            }
            if (bestJ < 0 || bestVal <= 0) break;
            visits[i].push_back({bestArr, bestJ});
            curT[i] = bestArr + o[bestJ]; curPos[i] = pt[bestJ];
            used[bestJ] = 1; // don't let two idle scopes both "claim" it pointlessly
        }
    }

    for (int i = 0; i < T; i++) {
        sort(visits[i].begin(), visits[i].end());
        cout << visits[i].size() << "\n";
        for (auto &p : visits[i]) cout << p.first << " " << p.second << "\n";
    }
    return 0;
}
