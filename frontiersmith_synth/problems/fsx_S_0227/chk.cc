#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Metro Busker Permit Scheduling (max-weight independent set on a
// general conflict graph). Maximization. Internal baseline B = value of the single
// highest-value pitch (a one-pitch set is always conflict-free, so B is achievable and
// positive). Participant value F = sum of values of the granted pitches, provided the
// granted set is conflict-free and the indices are distinct/in range.
// ratio = min(1, (F / B) / 10).
int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    long long M = inf.readLong();

    vector<long long> w(N + 1);
    long long B = 0;                        // best single pitch
    for (int j = 1; j <= N; j++) {
        w[j] = inf.readInt();
        if (w[j] > B) B = w[j];
    }

    vector<int> eu(M), ev(M);
    for (long long i = 0; i < M; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
    }

    // read participant schedule: k, then k distinct indices in [1,N]
    int k = ouf.readInt(0, N, "k");
    vector<char> sel(N + 1, 0);
    long long F = 0;
    for (int t = 0; t < k; t++) {
        int idx = ouf.readInt(1, N, "s");
        if (sel[idx])
            quitf(_wa, "pitch %d granted more than once", idx);
        sel[idx] = 1;
        F += w[idx];
    }
    if (!ouf.seekEof())
        quitf(_wa, "trailing tokens after the %d granted pitches", k);

    // conflict-free check: no incompatible pair may be fully granted
    for (long long i = 0; i < M; i++)
        if (sel[eu[i]] && sel[ev[i]])
            quitf(_wa, "granted pitches %d and %d conflict", eu[i], ev[i]);

    if (B <= 0)
        quitf(_fail, "internal: non-positive baseline B=%lld", B);

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
