#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int T, J, G;
vector<ll> sp, od;
vector<int> C;
vector<int> A, B_, W;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    T = inf.readInt();
    J = inf.readInt();
    G = inf.readInt();

    sp.assign(T, 0); od.assign(T, 0); C.assign(T, 0);
    for (int t = 0; t < T; t++) {
        sp[t] = inf.readInt();
        od[t] = inf.readInt();
        C[t]  = inf.readInt();
    }
    A.assign(J, 0); B_.assign(J, 0); W.assign(J, 0);
    for (int j = 0; j < J; j++) {
        A[j]  = inf.readInt();
        B_[j] = inf.readInt();
        W[j]  = inf.readInt();
    }

    // ---- internal trivial baseline B: earliest contiguous, all shore power ----
    ll Bcost = 0;
    for (int j = 0; j < J; j++) {
        Bcost += G;                                   // one contiguous run
        for (int t = A[j]; t < A[j] + W[j]; t++)
            Bcost += od[t];
    }
    if (Bcost <= 0) quitf(_fail, "bad instance: B=%lld", Bcost);

    // ---- read & validate participant schedule ----
    vector<int> spotUse(T, 0);
    ll F = 0;
    for (int j = 0; j < J; j++) {
        vector<char> used(T, 0);
        vector<int> hours;
        hours.reserve(W[j]);
        for (int k = 0; k < W[j]; k++) {
            int t = ouf.readInt(A[j], B_[j] - 1, "hour");
            if (used[t]) quitf(_wa, "ship %d works hour %d more than once", j + 1, t);
            used[t] = 1;
            int m = ouf.readInt(0, 1, "mode");
            if (m == 0) { spotUse[t]++; F += sp[t]; }
            else        {               F += od[t]; }
            hours.push_back(t);
        }
        // gantry charge: number of maximal consecutive-hour runs
        sort(hours.begin(), hours.end());
        int runs = 0;
        for (size_t i = 0; i < hours.size(); i++)
            if (i == 0 || hours[i] != hours[i - 1] + 1) runs++;
        F += (ll)G * runs;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // shared spot-capacity feasibility
    for (int t = 0; t < T; t++)
        if (spotUse[t] > C[t])
            quitf(_wa, "hour %d spot use %d exceeds capacity %d", t, spotUse[t], C[t]);

    if (F <= 0) quitf(_wa, "non-positive cost F=%lld", F);

    double sc = min(1000.0, 100.0 * (double)Bcost / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, Bcost, sc / 1000.0);
    return 0;
}
