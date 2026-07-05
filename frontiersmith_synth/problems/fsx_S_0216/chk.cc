#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int T = inf.readInt();
    int K = inf.readInt();
    vector<ll> vol(T + 1), D(T + 1), p(T + 1);
    for (int t = 1; t <= T; t++) vol[t] = inf.readInt();
    for (int t = 1; t <= T; t++) D[t] = inf.readInt();
    for (int t = 1; t <= T; t++) p[t] = inf.readInt();
    vector<int> L(K + 1), R(K + 1), W(K + 1);
    for (int i = 1; i <= K; i++) {
        L[i] = inf.readInt();
        R[i] = inf.readInt();
        W[i] = inf.readInt();
    }

    // ---- internal baseline B: contract-only on earliest W_i nights of each window ----
    {
        vector<ll> cntC(T + 1, 0);
        for (int i = 1; i <= K; i++)
            for (int j = 0; j < W[i]; j++) cntC[L[i] + j]++;
        ll B = 0;
        for (int t = 1; t <= T; t++) {
            if (cntC[t] > 0) B += D[t];
            B += cntC[t] * p[t];
        }
        if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

        // ---- read & validate participant schedule ----
        vector<ll> volUsed(T + 1, 0);
        vector<ll> cntCon(T + 1, 0);      // contract guards per night
        vector<int> nightSeg(T + 1, 0);   // which segment last claimed a night (dup check)
        for (int i = 1; i <= K; i++) {
            int g = ouf.readInt(W[i], R[i] - L[i] + 1, "g_i");
            for (int e = 0; e < g; e++) {
                int t = ouf.readInt(1, T, "night");
                if (t < L[i] || t > R[i])
                    quitf(_wa, "segment %d guarded on night %d outside window [%d,%d]",
                          i, t, L[i], R[i]);
                if (nightSeg[t] == i)
                    quitf(_wa, "segment %d guarded twice on night %d", i, t);
                nightSeg[t] = i;
                int m = ouf.readInt(0, 1, "mode");
                if (m == 0) volUsed[t]++;
                else cntCon[t]++;
            }
        }
        if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

        for (int t = 1; t <= T; t++)
            if (volUsed[t] > vol[t])
                quitf(_wa, "night %d uses %lld volunteers > available %lld",
                      t, volUsed[t], vol[t]);

        ll F = 0;
        for (int t = 1; t <= T; t++) {
            if (cntCon[t] > 0) F += D[t];
            F += cntCon[t] * p[t];
        }

        double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
        quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    }
    return 0;
}
