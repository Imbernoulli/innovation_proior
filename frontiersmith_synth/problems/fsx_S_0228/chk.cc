#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);
    long long T = inf.readLong();
    long long M = inf.readLong();
    long long S = inf.readLong();
    long long D = inf.readLong();
    long long P = inf.readLong();
    long long h = inf.readLong();
    long long H = inf.readLong();
    vector<long long> a(T);
    vector<int> av(T);
    for (long long i = 0; i < T; i++) {
        a[i] = inf.readLong();
        av[i] = (int)inf.readLong();
    }
    // read participant decisions
    vector<int> on(T), sp(T);
    for (long long i = 0; i < T; i++) {
        on[i] = ouf.readInt(0, 1, "on");
        sp[i] = ouf.readInt(0, 1, "spot");
        if (sp[i] == 1 && av[i] == 0)
            quitf(_wa, "spot lift run in slot %lld where avail=0", i + 1);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // replay
    long long q = 0, F = 0;
    for (long long i = 0; i < T; i++) {
        q += a[i];
        long long cap = (on[i] ? M : 0) + (sp[i] ? S : 0);
        long long served = min(q, cap);
        q -= served;
        F += (long long)on[i] * D + (long long)sp[i] * P + h * q;
    }
    F += H * q; // stranded overnight

    long long B = D * T; // baseline: run on-demand every slot (always empties -> no holding)
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
