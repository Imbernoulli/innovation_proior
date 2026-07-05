#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static long long manh(long long ax, long long ay, long long bx, long long by){
    return llabs(ax-bx) + llabs(ay-by);
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    // ---- read input ----
    int P = inf.readInt(1, 300, "P");
    int Q = inf.readInt(2, 30, "Q");
    long long x0 = inf.readInt(0, 10000, "x0");
    long long y0 = inf.readInt(0, 10000, "y0");

    vector<long long> px(P+1), py(P+1), dx(P+1), dy(P+1), q(P+1), w(P+1);
    long long B = 0; // serve-nobody baseline = sum of all penalties
    for (int i = 1; i <= P; i++) {
        px[i] = inf.readInt(0, 10000, "px");
        py[i] = inf.readInt(0, 10000, "py");
        dx[i] = inf.readInt(0, 10000, "dx");
        dy[i] = inf.readInt(0, 10000, "dy");
        q[i]  = inf.readInt(1, 8, "q");
        w[i]  = inf.readLong(1, 50000000LL, "w");
        B += w[i];
    }

    // ---- read participant output ----
    int M = ouf.readInt(0, 2*P, "M");

    vector<char> pickedUp(P+1, 0), delivered(P+1, 0);
    long long load = 0;
    long long H = 0; // haulage cost
    long long curx = x0, cury = y0;

    for (int e = 0; e < M; e++) {
        int t = ouf.readInt(0, 1, "t");
        int i = ouf.readInt(1, P, "i");
        long long tx, ty;
        if (t == 0) { // capture / pickup
            if (pickedUp[i]) quitf(_wa, "request %d captured more than once (event %d)", i, e+1);
            tx = px[i]; ty = py[i];
        } else {      // release / delivery
            if (!pickedUp[i]) quitf(_wa, "request %d released before capture (event %d)", i, e+1);
            if (delivered[i]) quitf(_wa, "request %d released more than once (event %d)", i, e+1);
            tx = dx[i]; ty = dy[i];
        }
        // travel from current point to this event point at current load
        long long d = manh(curx, cury, tx, ty);
        H += d * (1 + load);
        // apply event to load
        if (t == 0) { pickedUp[i] = 1; load += q[i]; if (load > Q) quitf(_wa, "capacity exceeded: load %lld > Q %d at event %d", load, Q, e+1); }
        else        { delivered[i] = 1; load -= q[i]; }
        curx = tx; cury = ty;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing data after %d events", M);

    // each request must be captured iff released
    long long skipPenalty = 0;
    for (int i = 1; i <= P; i++) {
        if (pickedUp[i] && !delivered[i])
            quitf(_wa, "request %d captured but never released", i);
        if (!pickedUp[i] && delivered[i])
            quitf(_wa, "request %d released but never captured", i); // (unreachable, guarded above)
        if (!pickedUp[i]) skipPenalty += w[i]; // unserved
    }
    // load must have returned to 0 (guaranteed by the captured<=>released invariant)
    if (load != 0) quitf(_wa, "internal: nonzero final load %lld", load);

    // closing leg back to depot at load 0 (load is 0 here); add for completeness
    H += manh(curx, cury, x0, y0) * (1 + load);

    long long F = H + skipPenalty;
    if (F < 1) F = 1; // safety; F>=1 always since B>=1 and serving costs >=0

    // minimization scoring convention
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc/1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc/1000.0);
    return 0;
}
