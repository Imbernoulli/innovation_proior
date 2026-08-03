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

    vector<ll> px(P + 1), py(P + 1), dx(P + 1), dy(P + 1), q(P + 1), c(P + 1), w(P + 1);
    ll B = 0;
    for (int i = 1; i <= P; i++) {
        px[i] = inf.readInt(); py[i] = inf.readInt();
        dx[i] = inf.readInt(); dy[i] = inf.readInt();
        q[i]  = inf.readInt(); c[i] = inf.readInt(); w[i] = inf.readLong();
        B += w[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    int L = ouf.readInt(0, 2 * P, "L");
    vector<int> mnt(P + 1, 0), rd(P + 1, 0);
    ll dist = 0, load = 0;
    ll cx = x0, cy = y0;
    bool any = false;

    for (int e = 0; e < L; e++) {
        int t = ouf.readInt(0, 1, "type");
        int i = ouf.readInt(1, P, "idx");
        if (t == 0) {
            if (mnt[i]) quitf(_wa, "task %d mounted more than once", i);
            mnt[i] = 1;
            load += q[i];
            if (load > Q) quitf(_wa, "rack capacity exceeded at event %d (load %lld > Q %d)", e, load, Q);
            dist += cheb(cx, cy, px[i], py[i]); cx = px[i]; cy = py[i];
        } else {
            if (!mnt[i]) quitf(_wa, "read of task %d before its mount", i);
            if (rd[i]) quitf(_wa, "task %d read more than once", i);
            rd[i] = 1;
            load -= q[i];
            dist += cheb(cx, cy, dx[i], dy[i]); cx = dx[i]; cy = dy[i];
        }
        any = true;
    }
    for (int i = 1; i <= P; i++)
        if (mnt[i] != rd[i]) quitf(_wa, "task %d is mounted but never read (or vice versa)", i);

    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (any) dist += cheb(cx, cy, x0, y0);

    ll instr = 0, pen = 0;
    for (int i = 1; i <= P; i++) {
        if (mnt[i]) instr += c[i];
        else        pen   += w[i];
    }

    ll F = dist + instr + pen;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
