// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// The obvious first attempt: EACH telescope, independently and with no
// notion of a partner, repeatedly walks to whichever reachable target
// currently offers it the highest decayed value, and commits (exact
// arrival, no idling). It never checks whether any other-site telescope
// will also be there -- so a same-site personal lure (irresistibly
// zero-travel, zero-decay) gets grabbed and wasted, and most planted
// cross-site rendezvous opportunities are never found because finding them
// requires knowing a SPECIFIC partner telescope's own geometry, not just
// "what looks good from here".

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

    for (int i = 0; i < T; i++) {
        int curT = 0, curPos = pos[i];
        vector<pair<int,int>> visits;
        int cap = 120;
        while ((int)visits.size() < cap) {
            int bestJ = -1, bestArr = -1;
            ll bestVal = -1;
            for (int j = 0; j < M; j++) {
                int travel = travelTicks(curPos, pt[j], speed[i]);
                int arr = curT + travel;
                if (arr + o[j] > H) continue;
                if (arr < a[j]) continue;
                ll val = decayVal(v[j], arr - a[j]);
                if (val > bestVal) { bestVal = val; bestJ = j; bestArr = arr; }
            }
            if (bestJ < 0 || bestVal <= 0) break;
            visits.push_back({bestArr, bestJ});
            curT = bestArr + o[bestJ];
            curPos = pt[bestJ];
        }
        cout << visits.size() << "\n";
        for (auto &p : visits) cout << p.first << " " << p.second << "\n";
    }
    return 0;
}
