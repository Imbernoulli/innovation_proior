#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Resonant Beacon Cover -- checker.
// Input:  q n r m M Kmax
//         m lines of H (n ints, 0..q-1)
//         M lines: n site symbols (0..q-1) + tolerance bit (0/1)
// Output: K, then K beacons (n symbols each, 0..q-1).
// A beacon c covers site i iff Hamming(c,x_i) <= r + t_i + res(c), where
// res(c)=1 iff H*c === 0 (mod q).
// Objective F = K (minimize). Internal baseline B = M (one beacon per site,
// planted exactly on the site -> distance 0 always covers).
// Score: ratio = min(1, 0.1 * B / F).
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int q = inf.readInt(2, 5, "q");
    int n = inf.readInt(1, 20, "n");
    int r = inf.readInt(0, 5, "r");
    int m = inf.readInt(0, 10, "m");
    int M = inf.readInt(1, 5000, "M");
    int Kmax = inf.readInt(1, 5000, "Kmax");

    vector<vector<int>> H(m, vector<int>(n));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            H[i][j] = inf.readInt(0, q - 1, "H_ij");

    vector<vector<int>> site(M, vector<int>(n));
    vector<int> tol(M);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n; j++) site[i][j] = inf.readInt(0, q - 1, "site_symbol");
        tol[i] = inf.readInt(0, 1, "tolerance");
    }

    long long K = ouf.readLong(1LL, (long long)Kmax, "K");
    vector<vector<int>> beacon(K, vector<int>(n));
    vector<int> res(K, 0);
    for (long long c = 0; c < K; c++) {
        for (int j = 0; j < n; j++) beacon[c][j] = ouf.readInt(0, q - 1, "beacon_symbol");
        long long s = 0;
        bool resonant = true;
        for (int row = 0; row < m && resonant; row++) {
            long long acc = 0;
            for (int j = 0; j < n; j++) acc += (long long)H[row][j] * beacon[c][j];
            if (acc % q != 0) resonant = false;
        }
        res[c] = resonant ? 1 : 0;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %lld beacons", K);

    for (int i = 0; i < M; i++) {
        bool covered = false;
        for (long long c = 0; c < K && !covered; c++) {
            int dist = 0;
            for (int j = 0; j < n; j++) if (beacon[c][j] != site[i][j]) dist++;
            int eff = r + tol[i] + res[c];
            if (dist <= eff) covered = true;
        }
        if (!covered) quitf(_wa, "site %d (0-indexed) is not covered by any of the %lld beacons", i, K);
    }

    long long F = K;
    long long B = M;
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    double ratio = sc / 1000.0;
    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, B, ratio);
    return 0;
}
