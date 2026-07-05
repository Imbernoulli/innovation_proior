#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int T, N, M;
ll G;
vector<ll> f, s;      // 1..T
vector<int> cap;      // 1..T
vector<int> r, dl;    // 0..N-1

// cost of a single step given how many units y are placed on it
static inline ll stepCost(int t, int y) {
    if (y <= 0) return 0;
    ll c = f[t];
    int sp = min((ll)y, (ll)cap[t]);
    c += s[t] * (ll)sp;
    c += G * (ll)(y - sp);
    return c;
}

static inline ll totalCost(const vector<int>& y) {
    ll tot = 0;
    for (int t = 1; t <= T; t++) tot += stepCost(t, y[t]);
    return tot;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    T = inf.readInt();
    N = inf.readInt();
    M = inf.readInt();
    G = inf.readInt();
    f.assign(T + 1, 0); s.assign(T + 1, 0); cap.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) {
        f[t] = inf.readInt();
        s[t] = inf.readInt();
        cap[t] = inf.readInt();
    }
    r.assign(N, 0); dl.assign(N, 0);
    for (int i = 0; i < N; i++) {
        r[i] = inf.readInt();
        dl[i] = inf.readInt();
    }

    // ---- internal baseline B: earliest-deadline-first pack (ignores cost) ----
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (dl[a] != dl[b]) return dl[a] < dl[b];
        if (r[a] != r[b]) return r[a] < r[b];
        return a < b;
    });
    vector<int> yb(T + 1, 0);
    bool feasibleInstance = true;
    for (int idx : order) {
        int placed = -1;
        for (int t = r[idx]; t <= dl[idx]; t++) {
            if (yb[t] < M) { placed = t; break; }
        }
        if (placed < 0) { feasibleInstance = false; break; }
        yb[placed]++;
    }
    if (!feasibleInstance) quitf(_fail, "bad instance: baseline EDF could not schedule all tasks");
    ll B = totalCost(yb);
    if (B <= 0) quitf(_fail, "bad instance: baseline cost B=%lld not positive", B);

    // ---- read & validate participant assignment ----
    vector<int> y(T + 1, 0);
    for (int i = 0; i < N; i++) {
        int t = ouf.readInt(1, T, "step");
        if (t < r[i] || t > dl[i])
            quitf(_wa, "task %d assigned to step %d outside window [%d,%d]", i + 1, t, r[i], dl[i]);
        y[t]++;
        if (y[t] > M)
            quitf(_wa, "step %d exceeds throughput M=%d", t, M);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = totalCost(y);
    if (F <= 0) quitf(_wa, "non-positive participant cost F=%lld", F);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
