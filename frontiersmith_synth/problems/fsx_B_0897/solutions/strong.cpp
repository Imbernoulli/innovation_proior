// TIER: strong
// The insight: an island's row- and column-stiffness are independent sums, and a
// single rod can only feed its CHEAP axial term into one of them -- the other
// axis is stuck with the rod's weak cubic bending term, which is what forces
// large widths on long spans. A pair of orthogonal rods (one vertical, one
// horizontal) lets EACH axis be satisfied by its own cheap axial term instead.
// For every island we evaluate both the best single-direction rod AND the best
// orthogonal pair (trying all 4 vertical/horizontal combinations, each sized to
// the true minimum feasible width) and keep whichever is cheaper under the real
// visual-cost map -- so on islands where Sreq is small enough that a thin single
// rod is already cheap, we still use it; the pair only wins where it should.
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
ll sumCost(ll r1, ll c1, ll r2, ll c2) {
    ll s = 0;
    for (ll r = r1; r <= r2; r++)
        for (ll c = c1; c <= c2; c++)
            s += Cst[r][c] - '0';
    return s;
}

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
        // --- best single-direction rod (must cover both axial own-axis and
        // bending cross-axis) ---
        ll bestSingleCost = LLONG_MAX;
        array<ll,5> bestSingleRect{};
        for (int d = UP; d <= RIGHT; d++) {
            ll L = gapLen(id, d);
            double wa = ceilDiv_(Sreq[id] * (double)L / A);
            double wb = ceilDiv_((double)L * cbrt(Sreq[id] * (double)BD));
            ll w = (ll)max((double)1, max(wa, wb));
            ll capW = maxWidth(id, d);
            if (w > capW) continue; // infeasible in this direction
            ll r1, c1, r2, c2; rectFor(id, d, L, w, r1, c1, r2, c2);
            ll cost = sumCost(r1, c1, r2, c2);
            int o = (d == UP || d == DOWN) ? 0 : 1;
            if (cost < bestSingleCost) { bestSingleCost = cost; bestSingleRect = {o, r1, c1, r2, c2}; }
        }
        // --- best orthogonal pair: one of {UP,DOWN} x one of {LEFT,RIGHT},
        // each sized via the cheap axial-only formula for its OWN axis ---
        ll bestPairCost = LLONG_MAX;
        array<ll,5> bestVRect{}, bestHRect{};
        for (int dv : {UP, DOWN}) {
            ll Lv = gapLen(id, dv);
            ll wv = (ll)max(1.0, (double)ceilDiv_(Sreq[id] * (double)Lv / A));
            if (wv > maxWidth(id, dv)) continue;
            ll vr1, vc1, vr2, vc2; rectFor(id, dv, Lv, wv, vr1, vc1, vr2, vc2);
            ll vcost = sumCost(vr1, vc1, vr2, vc2);
            for (int dh : {LEFT, RIGHT}) {
                ll Lh = gapLen(id, dh);
                ll wh = (ll)max(1.0, (double)ceilDiv_(Sreq[id] * (double)Lh / A));
                if (wh > maxWidth(id, dh)) continue;
                ll hr1, hc1, hr2, hc2; rectFor(id, dh, Lh, wh, hr1, hc1, hr2, hc2);
                ll hcost = sumCost(hr1, hc1, hr2, hc2);
                ll total = vcost + hcost;
                if (total < bestPairCost) {
                    bestPairCost = total;
                    bestVRect = {0, vr1, vc1, vr2, vc2};
                    bestHRect = {1, hr1, hc1, hr2, hc2};
                }
            }
        }
        if (bestPairCost < bestSingleCost && bestPairCost < LLONG_MAX) {
            bridges.push_back(bestVRect);
            bridges.push_back(bestHRect);
        } else {
            bridges.push_back(bestSingleRect);
        }
    }
    cout << bridges.size() << "\n";
    for (auto &b : bridges) cout << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << " " << b[4] << "\n";
    return 0;
}
