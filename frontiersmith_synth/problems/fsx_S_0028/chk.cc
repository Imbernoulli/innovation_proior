#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static inline long long edist(long long x1, long long y1, long long x2, long long y2) {
    double dx = (double)(x1 - x2), dy = (double)(y1 - y2);
    return (long long)llround(sqrt(dx * dx + dy * dy));
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    // ---- read input ----
    int N = inf.readInt();
    int Q = inf.readInt();
    long long x0 = inf.readInt();
    long long y0 = inf.readInt();
    vector<long long> px(N + 1), py(N + 1), dx(N + 1), dy(N + 1);
    vector<int> q(N + 1);
    for (int i = 1; i <= N; i++) {
        px[i] = inf.readInt(); py[i] = inf.readInt();
        dx[i] = inf.readInt(); dy[i] = inf.readInt();
        q[i]  = inf.readInt();
    }

    // ---- read participant output ----
    int L = ouf.readInt(0, 2 * N, "L");
    vector<int> pickedAt(N + 1, -1), deliveredAt(N + 1, -1);
    vector<pair<int,int>> events(L); // (type, job)

    long long load = 0;
    for (int e = 0; e < L; e++) {
        int t = ouf.readInt(0, 1, "t");
        int i = ouf.readInt(1, N, "job");
        events[e] = {t, i};
        if (t == 0) {
            if (pickedAt[i] != -1)
                quitf(_wa, "job %d picked up more than once", i);
            pickedAt[i] = e;
            load += q[i];
            if (load > Q)
                quitf(_wa, "capacity exceeded at event %d: load %lld > Q %d", e, load, Q);
        } else {
            if (deliveredAt[i] != -1)
                quitf(_wa, "job %d delivered more than once", i);
            if (pickedAt[i] == -1)
                quitf(_wa, "job %d delivered before pickup", i);
            deliveredAt[i] = e;
            load -= q[i];
            if (load < 0)
                quitf(_wa, "negative load at event %d (internal)", e);
        }
    }
    if (!ouf.seekEof())
        quitf(_wa, "trailing output after %d events", L);

    // mandatory service: every job served exactly once each
    for (int i = 1; i <= N; i++) {
        if (pickedAt[i] == -1)
            quitf(_wa, "job %d never picked up (service is mandatory)", i);
        if (deliveredAt[i] == -1)
            quitf(_wa, "job %d never delivered (service is mandatory)", i);
    }

    // ---- objective F: realized tour length ----
    long long F = 0;
    long long cx = x0, cy = y0;
    for (int e = 0; e < L; e++) {
        int t = events[e].first, i = events[e].second;
        long long nx = (t == 0) ? px[i] : dx[i];
        long long ny = (t == 0) ? py[i] : dy[i];
        F += edist(cx, cy, nx, ny);
        cx = nx; cy = ny;
    }
    F += edist(cx, cy, x0, y0);

    // ---- baseline B: serialized plan (pickup i, deliver i, in index order) ----
    long long B = 0;
    cx = x0; cy = y0;
    for (int i = 1; i <= N; i++) {
        B += edist(cx, cy, px[i], py[i]);
        cx = px[i]; cy = py[i];
        B += edist(cx, cy, dx[i], dy[i]);
        cx = dx[i]; cy = dy[i];
    }
    B += edist(cx, cy, x0, y0);
    if (B < 1) B = 1; // guard: keep baseline positive

    // ---- score (minimization) ----
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
