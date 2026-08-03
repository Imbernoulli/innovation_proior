#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for satellite ground-station activation (max-weight independent set).
// Maximization. Baseline B = value of the single highest-value station (a one-station set
// is always conflict-free, so B is achievable and positive). Participant value F = sum of
// values of the activated stations, provided the activated set is conflict-free and the
// indices are distinct/in range.  ratio = min(1, (F / B) / 10).
int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();

    vector<long long> w(N + 1);
    long long B = 0;                       // best single station
    for (int j = 1; j <= N; j++) {
        w[j] = inf.readInt();
        if (w[j] > B) B = w[j];
    }

    vector<int> eu(M), ev(M);
    for (int i = 0; i < M; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
    }

    // read participant activation set: k, then k distinct indices in [1,N]
    int k = ouf.readInt(0, N, "k");
    vector<char> sel(N + 1, 0);
    long long F = 0;
    for (int t = 0; t < k; t++) {
        int idx = ouf.readInt(1, N, "s");
        if (sel[idx])
            quitf(_wa, "station %d activated more than once", idx);
        sel[idx] = 1;
        F += w[idx];
    }
    if (!ouf.seekEof())
        quitf(_wa, "trailing tokens after the %d activated stations", k);

    // conflict-free check: no interfering pair may be fully activated
    for (int i = 0; i < M; i++)
        if (sel[eu[i]] && sel[ev[i]])
            quitf(_wa, "activated stations %d and %d interfere", eu[i], ev[i]);

    if (B <= 0)
        quitf(_fail, "internal: non-positive baseline B=%lld", B);

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
