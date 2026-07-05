#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();

    vector<int> pk(m);
    vector<vector<pair<int,int>>> pat(m);
    vector<long long> pw(m);
    long long B = 0;

    for (int j = 0; j < m; j++) {
        int k = inf.readInt();
        pk[j] = k;
        pat[j].resize(k);
        bool allzero = true;
        for (int e = 0; e < k; e++) {
            int s = inf.readInt(1, n, "station");
            int o = inf.readInt(0, 1, "orientation_req");
            pat[j][e] = {s, o};
            if (o != 0) allzero = false;
        }
        long long w = inf.readInt();
        pw[j] = w;
        if (allzero) B += w;
    }

    // participant assignment: exactly n bits, nothing else.
    vector<int> b(n + 1, 0);
    for (int i = 1; i <= n; i++)
        b[i] = ouf.readInt(0, 1, "alignment");
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d alignments", n);

    long long F = 0;
    for (int j = 0; j < m; j++) {
        bool ok = true;
        for (auto& pr : pat[j])
            if (b[pr.first] != pr.second) { ok = false; break; }
        if (ok) F += pw[j];
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
