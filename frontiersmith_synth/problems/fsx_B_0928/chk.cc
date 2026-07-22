#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static const double EPS = 1e-6;

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int W = inf.readInt(10, 400, "W");
    int H = inf.readInt(10, 400, "H");
    int K = inf.readInt(3, 9, "K");
    vector<long long> T(K + 1);
    for (int i = 1; i <= K; i++) T[i] = inf.readInt(0, 2000000000, "T");
    int N = inf.readInt(1, 500, "N");
    vector<long long> r(N + 1), v(N + 1), arr(N + 1), dep(N + 1);
    for (int i = 1; i <= N; i++) {
        r[i] = inf.readInt(1, 2000000000, "r");
        v[i] = inf.readInt(1, 1000000, "v");
        arr[i] = inf.readInt(0, 2000000000, "arr");
        dep[i] = inf.readInt(0, 2000000000, "dep");
    }

    // internal baseline B: the best single ship that could be anchored alone.
    long long B = 0;
    for (int i = 1; i <= N; i++) {
        if (2 * r[i] > (long long) min(W, H)) continue;
        int a = -1, b = -1;
        for (int t = 1; t <= K; t++) if (T[t] >= arr[i]) { a = t; break; }
        for (int t = K; t >= 1; t--) if (T[t] <= dep[i]) { b = t; break; }
        if (a != -1 && b != -1 && a < b) B = max(B, v[i]);
    }
    if (B <= 0) quitf(_fail, "internal: generator produced no feasible single-ship baseline");

    int M = ouf.readInt(0, N, "M");
    vector<char> used(N + 1, 0);
    struct Acc { double x, y; int a, b; long long r; };
    vector<Acc> accepted;
    accepted.reserve(M);
    long long F = 0;

    for (int k = 0; k < M; k++) {
        int idx = ouf.readInt(1, N, "idx");
        if (used[idx]) quitf(_wa, "ship %d listed more than once in the output", idx);
        used[idx] = 1;

        double x = ouf.readDouble(-1e7, 1e7, "x");
        double y = ouf.readDouble(-1e7, 1e7, "y");
        if (!isfinite(x) || !isfinite(y)) quitf(_wa, "ship %d: non-finite anchor coordinate", idx);

        int a = ouf.readInt(1, K, "a");
        int b = ouf.readInt(1, K, "b");
        if (a >= b) quitf(_wa, "ship %d: entry tick index %d not strictly before exit tick index %d", idx, a, b);

        if (T[a] < arr[idx] - EPS)
            quitf(_wa, "ship %d: enters at tide %lld before its arrival window %lld", idx, T[a], arr[idx]);
        if (T[b] > dep[idx] + EPS)
            quitf(_wa, "ship %d: leaves at tide %lld after its departure deadline %lld", idx, T[b], dep[idx]);

        double rr = (double) r[idx];
        if (x < rr - EPS || x > (double) W - rr + EPS || y < rr - EPS || y > (double) H - rr + EPS)
            quitf(_wa, "ship %d: swing circle (centre (%.4f,%.4f), r=%.4f) leaves the %dx%d basin", idx, x, y, rr, W, H);

        for (auto &A : accepted) {
            if (max(a, A.a) < min(b, A.b)) {
                double dx = x - A.x, dy = y - A.y;
                double need = rr + A.r;
                if (dx * dx + dy * dy < need * need - 1e-6)
                    quitf(_wa, "ship %d's swing circle overlaps another anchored ship's circle during a shared epoch", idx);
            }
        }

        accepted.push_back({x, y, a, b, r[idx]});
        F += v[idx];
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the declared %d output lines", M);

    double sc = min(1000.0, 100.0 * (double) F / (double) max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
