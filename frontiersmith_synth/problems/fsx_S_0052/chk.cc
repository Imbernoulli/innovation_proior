#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll md(ll ax, ll ay, ll bx, ll by) {
    return llabs(ax - bx) + llabs(ay - by);
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int Q = inf.readInt();
    int M = inf.readInt();
    ll X0 = inf.readInt(), Y0 = inf.readInt();

    vector<ll> ax(N + 1), ay(N + 1), bx(N + 1), by(N + 1), q(N + 1);
    for (int i = 1; i <= N; i++) {
        ax[i] = inf.readInt(); ay[i] = inf.readInt();
        bx[i] = inf.readInt(); by[i] = inf.readInt();
        q[i]  = inf.readInt();
    }

    // ---- internal baseline B: serve every order, one cut at a time, in input order ----
    ll B = 0, cx = X0, cy = Y0;
    for (int i = 1; i <= N; i++) {
        B += md(cx, cy, ax[i], ay[i]); cx = ax[i]; cy = ay[i];
        B += md(cx, cy, bx[i], by[i]); cx = bx[i]; cy = by[i];
    }
    B += md(cx, cy, X0, Y0);
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld <= 0", B);

    // ---- read & validate the participant's shunting tour ----
    int L = ouf.readInt(0, 2 * N, "L");
    vector<int> pk(N + 1, 0), dl(N + 1, 0);
    ll dist = 0, load = 0;
    cx = X0; cy = Y0;

    for (int e = 0; e < L; e++) {
        int t = ouf.readInt(0, 1, "type");
        int j = ouf.readInt(1, N, "idx");
        if (t == 0) {
            if (pk[j]) quitf(_wa, "order %d picked up more than once", j);
            pk[j] = 1;
            load += q[j];
            if (load > Q) quitf(_wa, "drawbar limit exceeded at move %d (load %lld > Q %d)", e, load, Q);
            dist += md(cx, cy, ax[j], ay[j]); cx = ax[j]; cy = ay[j];
        } else {
            if (!pk[j]) quitf(_wa, "delivery of order %d before its pickup", j);
            if (dl[j]) quitf(_wa, "order %d delivered more than once", j);
            dl[j] = 1;
            load -= q[j];
            dist += md(cx, cy, bx[j], by[j]); cx = bx[j]; cy = by[j];
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    int done = 0;
    for (int i = 1; i <= N; i++) {
        if (pk[i] != dl[i]) quitf(_wa, "order %d is picked up but never delivered (or vice versa)", i);
        if (pk[i] && dl[i]) done++;
    }
    if (load != 0) quitf(_wa, "loco returns with cuts still on board (load %lld)", load);
    if (done < M) quitf(_wa, "only %d orders completed, quota M=%d not met", done, M);

    if (L > 0) dist += md(cx, cy, X0, Y0);   // close the tour back at the hump depot

    ll F = dist;
    if (F <= 0) quitf(_wa, "degenerate tour F=%lld", F);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
