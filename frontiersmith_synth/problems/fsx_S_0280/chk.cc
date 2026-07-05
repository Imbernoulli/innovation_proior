#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll DST(ll x1, ll y1, ll x2, ll y2) {
    ll dx = x1 - x2, dy = y1 - y2;
    return (ll)llround(sqrt((double)(dx * dx + dy * dy)));
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt(1, 300000, "N");
    ll SX = inf.readInt(0, 10000, "SX");
    ll SY = inf.readInt(0, 10000, "SY");
    vector<ll> px(N + 1), py(N + 1), dx(N + 1), dy(N + 1), w(N + 1);
    ll B = 0;
    for (int i = 1; i <= N; i++) {
        px[i] = inf.readInt(0, 10000, "px");
        py[i] = inf.readInt(0, 10000, "py");
        dx[i] = inf.readInt(0, 10000, "dx");
        dy[i] = inf.readInt(0, 10000, "dy");
        w[i] = inf.readInt(1, 1000000, "w");
        B += w[i];
    }

    // ---- read participant circuit ----
    int L = ouf.readInt(0, 2 * N, "L");
    vector<int> posP(N + 1, -1), posD(N + 1, -1);
    vector<int> seq(L);
    for (int k = 0; k < L; k++) {
        int v = ouf.readInt(-N, N, "event");
        if (v == 0) quitf(_wa, "event %d is zero (not a valid job event)", k);
        int j = abs(v);
        if (v > 0) {
            if (posP[j] != -1) quitf(_wa, "job %d picked up more than once", j);
            posP[j] = k;
        } else {
            if (posD[j] != -1) quitf(_wa, "job %d delivered more than once", j);
            posD[j] = k;
        }
        seq[k] = v;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d events", L);

    // ---- feasibility + unserved penalties ----
    ll unserved = 0;
    for (int j = 1; j <= N; j++) {
        bool hp = (posP[j] != -1), hd = (posD[j] != -1);
        if (hp != hd)
            quitf(_wa, "job %d is partially routed (needs both pickup and delivery, or neither)", j);
        if (hp && hd && posP[j] >= posD[j])
            quitf(_wa, "job %d: delivery precedes pickup (precedence violated)", j);
        if (!hp) unserved += w[j];
    }

    // ---- tour length ----
    ll cx = SX, cy = SY, tour = 0;
    for (int k = 0; k < L; k++) {
        int v = seq[k], j = abs(v);
        ll nx = (v > 0) ? px[j] : dx[j];
        ll ny = (v > 0) ? py[j] : dy[j];
        tour += DST(cx, cy, nx, ny);
        cx = nx; cy = ny;
    }
    tour += DST(cx, cy, SX, SY);

    ll F = tour + unserved;                 // objective (minimize)
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld tour=%lld unserved=%lld Ratio: %.6f",
          F, B, tour, unserved, sc / 1000.0);
    return 0;
}
