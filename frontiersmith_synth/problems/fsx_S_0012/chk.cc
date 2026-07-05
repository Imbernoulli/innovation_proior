#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int T = inf.readInt();
    long long D = inf.readInt();
    long long OD = inf.readInt();

    vector<long long> cap(T + 1), sc(T + 1);
    for (int s = 1; s <= T; s++) {
        cap[s] = inf.readInt();
        sc[s] = inf.readInt();
    }
    vector<long long> W(N + 1), d(N + 1);
    long long B = 0;
    for (int i = 1; i <= N; i++) {
        W[i] = inf.readInt();
        d[i] = inf.readInt();
        // reference all-on-demand cost: ceil(W_i / D) on-demand steps
        long long steps = (W[i] + D - 1) / D;
        B += OD * steps;
    }

    // Read participant schedule: exactly T lines "mode sec".
    const long long INF = (long long)4e18;
    vector<long long> prog(N + 1, 0);
    vector<long long> doneStep(N + 1, INF);
    long long F = 0;

    for (int s = 1; s <= T; s++) {
        int mode = ouf.readInt(0, 2, "mode");
        int sec = ouf.readInt(0, N, "sec");
        if (mode == 0) {
            if (sec != 0) quitf(_wa, "step %d: PAUSE must have sec=0, got %d", s, sec);
            // no progress, no cost
        } else {
            if (sec < 1 || sec > N) quitf(_wa, "step %d: sec out of range %d", s, sec);
            long long delta = (mode == 1) ? cap[s] : D;
            long long cost = (mode == 1) ? sc[s] : OD;
            prog[sec] += delta;
            F += cost;
            if (doneStep[sec] == INF && prog[sec] >= W[sec]) doneStep[sec] = s;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d steps", T);

    // Feasibility: every section complete by its deadline.
    for (int i = 1; i <= N; i++) {
        if (doneStep[i] > d[i]) {
            quitf(_wa, "section %d incomplete by deadline %lld (progress %lld/%lld, done@%lld)",
                  i, d[i], prog[i], W[i],
                  (doneStep[i] == INF ? (long long)-1 : doneStep[i]));
        }
    }

    if (B <= 0) quitf(_fail, "internal baseline non-positive");
    if (F <= 0) quitf(_wa, "zero-cost schedule cannot be feasible");

    double sc_score = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc_score / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc_score / 1000.0);
    return 0;
}
