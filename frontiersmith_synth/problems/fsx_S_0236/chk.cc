#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    vector<int> eu(m), ev(m), et(m);
    vector<long long> ew(m);
    for (int e = 0; e < m; e++) {
        eu[e] = inf.readInt();
        ev[e] = inf.readInt();
        et[e] = inf.readInt();
        ew[e] = inf.readInt();
    }

    // read participant output: n cryostat labels in {0,1}
    vector<int> side(n + 1);
    long long ones = 0;
    for (int i = 1; i <= n; i++) {
        int c = ouf.readInt(0, 1, format("c[%d]", i).c_str());
        side[i] = c;
        ones += c;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d labels", n);

    if (ones != n / 2)
        quitf(_wa, "unbalanced split: %lld modules in cryostat 1, expected %d", ones, n / 2);

    // earned-value objective for a given labeling
    auto earned = [&](const vector<int>& lab) -> long long {
        long long f = 0;
        for (int e = 0; e < m; e++) {
            bool cut = (lab[eu[e]] != lab[ev[e]]);
            bool sat = (et[e] == 0) ? cut : !cut; // coupling wants cut, bus wants uncut
            if (sat) f += ew[e];
        }
        return f;
    };

    long long F = earned(side);

    // internal baseline B: reference split modules 1..n/2 -> cryostat 0, rest -> cryostat 1
    vector<int> ref(n + 1);
    for (int i = 1; i <= n; i++) ref[i] = (i <= n / 2) ? 0 : 1;
    long long B = earned(ref);
    if (B <= 0) B = 1; // safety; generator guarantees a positive-value crossing wire

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
