// TIER: trivial
// Reproduces the checker's internal baseline: for every island, find the
// geometrically NEAREST single direction (tie-break up/down/left/right) and
// bridge it with ONE generously oversized (1.4x safety margin) rod. Ignores the
// axial/bending distinction entirely -- an unimaginative, safety-first build.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int R, C, K, A, BD;
vector<string> S, Cst;
vector<ll> ir1, ir2, ic1, ic2;
vector<double> Sreq;

enum Dir { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };
ll gapLen(int id, int dir) {
    if (dir == UP) return ir1[id] - 2;
    if (dir == DOWN) return R - ir2[id] - 1;
    if (dir == LEFT) return ic1[id] - 2;
    return C - ic2[id] - 1;
}
ll maxWidth(int id, int dir) {
    if (dir == UP || dir == DOWN) return ic2[id] - ic1[id] + 1;
    return ir2[id] - ir1[id] + 1;
}
void rectFor(int id, int dir, ll L, ll w, ll &r1, ll &c1, ll &r2, ll &c2) {
    if (dir == UP) { r1 = ir1[id] - L; r2 = ir1[id] - 1; ll iw = ic2[id] - ic1[id] + 1; c1 = ic1[id] + (iw - w) / 2; c2 = c1 + w - 1; }
    else if (dir == DOWN) { r1 = ir2[id] + 1; r2 = ir2[id] + L; ll iw = ic2[id] - ic1[id] + 1; c1 = ic1[id] + (iw - w) / 2; c2 = c1 + w - 1; }
    else if (dir == LEFT) { c1 = ic1[id] - L; c2 = ic1[id] - 1; ll ih = ir2[id] - ir1[id] + 1; r1 = ir1[id] + (ih - w) / 2; r2 = r1 + w - 1; }
    else { c1 = ic2[id] + 1; c2 = ic2[id] + L; ll ih = ir2[id] - ir1[id] + 1; r1 = ir1[id] + (ih - w) / 2; r2 = r1 + w - 1; }
}
ll ceilDiv_(double x) { return (ll)ceil(x - 1e-9); }

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    cin >> R >> C >> K >> A >> BD;
    S.assign(R + 1, "");
    for (int r = 1; r <= R; r++) { string row; cin >> row; S[r] = " " + row; }
    Cst.assign(R + 1, "");
    for (int r = 1; r <= R; r++) { string row; cin >> row; Cst[r] = " " + row; }
    ir1.assign(K + 1, INT_MAX); ir2.assign(K + 1, -1);
    ic1.assign(K + 1, INT_MAX); ic2.assign(K + 1, -1);
    for (int r = 1; r <= R; r++)
        for (int c = 1; c <= C; c++) {
            char ch = S[r][c];
            if (ch >= '1' && ch <= '9') {
                int id = ch - '0';
                ir1[id] = min(ir1[id], (ll)r); ir2[id] = max(ir2[id], (ll)r);
                ic1[id] = min(ic1[id], (ll)c); ic2[id] = max(ic2[id], (ll)c);
            }
        }
    Sreq.assign(K + 1, 0.0);
    for (int i = 1; i <= K; i++) { int id; double s; cin >> id >> s; Sreq[id] = s; }

    vector<array<ll,5>> bridges; // o r1 c1 r2 c2
    for (int id = 1; id <= K; id++) {
        int bestDir = UP; ll bestL = gapLen(id, UP);
        for (int d = DOWN; d <= RIGHT; d++) { ll g = gapLen(id, d); if (g < bestL) { bestL = g; bestDir = d; } }
        double Lm = (double)bestL;
        ll wa = ceilDiv_(Sreq[id] * Lm / A);
        ll wb = ceilDiv_(Lm * cbrt(Sreq[id] * (double)BD));
        ll singleW = max((ll)1, max(wa, wb));
        ll wTriv = max((ll)1, ceilDiv_(1.4 * (double)singleW));
        ll capW = maxWidth(id, bestDir);
        if (wTriv > capW) wTriv = capW;
        ll r1, c1, r2, c2;
        rectFor(id, bestDir, bestL, wTriv, r1, c1, r2, c2);
        int o = (bestDir == UP || bestDir == DOWN) ? 0 : 1;
        bridges.push_back({o, r1, c1, r2, c2});
    }
    cout << bridges.size() << "\n";
    for (auto &b : bridges) cout << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << " " << b[4] << "\n";
    return 0;
}
