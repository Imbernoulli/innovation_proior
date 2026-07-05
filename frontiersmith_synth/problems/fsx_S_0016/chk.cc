#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll cheb(ll ax, ll ay, ll bx, ll by) {
    return max(llabs(ax - bx), llabs(ay - by));
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int P = inf.readInt();
    int Q = inf.readInt();
    ll x0 = inf.readInt(), y0 = inf.readInt();

    vector<ll> px(P + 1), py(P + 1), dx(P + 1), dy(P + 1), q(P + 1), w(P + 1), f(P + 1);
    ll B = 0;
    for (int i = 1; i <= P; i++) {
        px[i] = inf.readInt(); py[i] = inf.readInt();
        dx[i] = inf.readInt(); dy[i] = inf.readInt();
        q[i]  = inf.readInt(); w[i]  = inf.readLong(); f[i] = inf.readLong();
        B += w[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    int L = ouf.readInt(0, 2 * P, "L");
    vector<int> pkPos(P + 1, -1), dlPos(P + 1, -1);
    ll dist = 0, load = 0;
    ll cx = x0, cy = y0;
    bool any = false;

    for (int e = 0; e < L; e++) {
        int t = ouf.readInt(0, 1, "type");
        int i = ouf.readInt(1, P, "idx");
        if (t == 0) {
            if (pkPos[i] != -1) quitf(_wa, "request %d picked up more than once", i);
            pkPos[i] = e;
            load += q[i];
            if (load > Q) quitf(_wa, "capacity exceeded at event %d (load %lld > Q %d)", e, load, Q);
            dist += cheb(cx, cy, px[i], py[i]); cx = px[i]; cy = py[i];
        } else {
            if (pkPos[i] == -1) quitf(_wa, "delivery of request %d before its pickup", i);
            if (dlPos[i] != -1) quitf(_wa, "request %d delivered more than once", i);
            dlPos[i] = e;
            load -= q[i];
            dist += cheb(cx, cy, dx[i], dy[i]); cx = dx[i]; cy = dy[i];
        }
        any = true;
    }
    for (int i = 1; i <= P; i++)
        if ((pkPos[i] == -1) != (dlPos[i] == -1))
            quitf(_wa, "request %d is picked up but never delivered (or vice versa)", i);

    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (any) dist += cheb(cx, cy, x0, y0);

    ll carry = 0, pen = 0;
    for (int i = 1; i <= P; i++) {
        if (pkPos[i] != -1) {
            ll hops = (ll)dlPos[i] - (ll)pkPos[i] - 1;   // stops made while trays i ride
            carry += f[i] * hops;
        } else {
            pen += w[i];
        }
    }

    ll F = dist + carry + pen;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld dist=%lld carry=%lld pen=%lld Ratio: %.6f",
          F, B, dist, carry, pen, sc / 1000.0);
    return 0;
}
