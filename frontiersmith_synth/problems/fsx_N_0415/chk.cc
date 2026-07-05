#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll L1(ll ax, ll ay, ll bx, ll by) {
    return llabs(ax - bx) + llabs(ay - by);
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int W = inf.readInt();
    int H = inf.readInt();
    int m = inf.readInt();
    int R = inf.readInt();
    int D = inf.readInt();
    int A = inf.readInt();
    int Bc = inf.readInt();
    int K = inf.readInt();
    int r0 = inf.readInt();
    vector<int> dx(D), dy(D);
    for (int i = 0; i < D; i++) { dx[i] = inf.readInt(); dy[i] = inf.readInt(); }

    auto resonant = [&](ll x, ll y) -> bool {
        ll v = ((A * x + Bc * y) % K + K) % K;
        return v == r0;
    };

    // ---- read participant output: exactly m distinct in-range pods ----
    vector<int> px(m), py(m);
    unordered_set<ll> seen; seen.reserve(m * 2 + 16);
    for (int i = 0; i < m; i++) {
        int x = ouf.readInt(0, W - 1, "pod_x");
        int y = ouf.readInt(0, H - 1, "pod_y");
        ll key = (ll)x * (ll)H + (ll)y;
        if (!seen.insert(key).second) quitf(_wa, "duplicate pod at (%d,%d)", x, y);
        px[i] = x; py[i] = y;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d pods", m);

    // ---- coverage (hard constraint) ----
    for (int j = 0; j < D; j++) {
        ll best = LLONG_MAX;
        for (int i = 0; i < m; i++) {
            ll dd = L1(dx[j], dy[j], px[i], py[i]);
            if (dd < best) { best = dd; if (best == 0) break; }
        }
        if (best > R)
            quitf(_wa, "demand %d (%d,%d) uncovered: nearest L1=%lld > R=%d", j, dx[j], dy[j], best, R);
    }

    // ---- objective F ----
    ll Dmin = LLONG_MAX;
    for (int i = 0; i < m; i++)
        for (int k = i + 1; k < m; k++) {
            ll dd = L1(px[i], py[i], px[k], py[k]);
            if (dd < Dmin) Dmin = dd;
        }
    if (m < 2 || Dmin == LLONG_MAX) Dmin = 0;
    ll Align = 0;
    for (int i = 0; i < m; i++) if (resonant(px[i], py[i])) Align++;
    double F = (double)Dmin * (1.0 + (double)Align / (double)m);

    // ---- internal reference layout B (deterministic, always feasible) ----
    // greedy cover in input order (hubs first) then fill spares by farthest-point over
    // the DEMAND cells (a moderate reference confined to the demand region).
    vector<ll> mind(D, LLONG_MAX);
    vector<char> used(D, 0);
    vector<int> bx, by; bx.reserve(m); by.reserve(m);
    int placed = 0;
    auto place = [&](int idx) {
        used[idx] = 1; placed++; bx.push_back(dx[idx]); by.push_back(dy[idx]);
        for (int j = 0; j < D; j++) {
            ll dd = L1(dx[idx], dy[idx], dx[j], dy[j]);
            if (dd < mind[j]) mind[j] = dd;
        }
    };
    for (int i = 0; i < D && placed < m; i++) if (mind[i] > R) place(i);           // cover
    while (placed < m) {                                                            // farthest-point spares
        int best = -1; ll bv = -1;
        for (int j = 0; j < D; j++) { if (used[j]) continue; if (mind[j] > bv) { bv = mind[j]; best = j; } }
        if (best < 0) break;
        place(best);
    }

    ll DminB = LLONG_MAX;
    int bcnt = (int)bx.size();
    for (int i = 0; i < bcnt; i++)
        for (int k = i + 1; k < bcnt; k++) {
            ll dd = L1(bx[i], by[i], bx[k], by[k]);
            if (dd < DminB) DminB = dd;
        }
    if (bcnt < 2 || DminB == LLONG_MAX) DminB = 1;
    if (DminB < 1) DminB = 1;
    ll AlignB = 0;
    for (int i = 0; i < bcnt; i++) if (resonant(bx[i], by[i])) AlignB++;
    double B = (double)DminB * (1.0 + (double)AlignB / (double)max(1, bcnt));
    if (B < 1.0) B = 1.0;

    double sc = min(1000.0, 100.0 * F / max(1.0, B));
    quitp(sc / 1000.0, "OK F=%.3f B=%.3f Dmin=%lld Align=%lld DminB=%lld AlignB=%lld Ratio: %.6f",
          F, B, Dmin, Align, DminB, AlignB, sc / 1000.0);
    return 0;
}
