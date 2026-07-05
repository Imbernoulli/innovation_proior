#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    if (t < 1) t = 1;
    if (t > 10) t = 10;

    // Difficulty ladder: patches and foragers grow; reachability widens.
    int Ps[11] = {0,   3,   5,   8,  15,  30,  60, 100, 160, 220, 280};
    int Bs[11] = {0,   4,  12,  30,  80, 200, 500,1000,1800,2800,4000};
    int mMx[11]= {0,   2,   3,   3,   4,   5,   6,   6,   7,   8,   8};
    int P = Ps[t], B = Bs[t];
    int mmax = min(P, mMx[t]);
    int mmin = min(P, 2);

    // Nectar stocks: sized so that concentrating foragers on a patch saturates it,
    // but spreading them recovers a lot -> big room between reference and optimum.
    // "target" ~ typical per-patch appetite load; stocks sit around 0.4..1.1 of it.
    double avgAppetite = 55.0;
    double avgM = 0.5 * (mmin + mmax);
    double target = (double)B * avgM / (double)P * avgAppetite;
    long long lo = (long long)max(60.0, 0.35 * target);
    long long hi = (long long)max(120.0, 1.10 * target);
    if (hi < lo + 40) hi = lo + 40;
    if (hi > 200000) hi = 200000;
    if (lo > hi) lo = hi;

    vector<long long> S(P);
    for (int j = 0; j < P; j++) S[j] = rnd.next(lo, hi);

    // A persistent permutation of patch indices; partial Fisher-Yates per forager
    // keeps it a valid permutation so no restore is needed.
    vector<int> idx(P);
    for (int j = 0; j < P; j++) idx[j] = j + 1;

    // Build forager lines into a buffer first (single write at the end).
    string out;
    out.reserve((size_t)B * 24 + P * 8 + 32);
    char buf[64];

    snprintf(buf, sizeof(buf), "%d %d\n", P, B);
    out += buf;
    for (int j = 0; j < P; j++) {
        snprintf(buf, sizeof(buf), "%lld%c", S[j], j + 1 == P ? '\n' : ' ');
        out += buf;
    }

    for (int i = 0; i < B; i++) {
        int m = rnd.next(mmin, mmax);
        // pick m distinct patches via partial shuffle
        for (int k = 0; k < m; k++) {
            int r = rnd.next(k, P - 1);
            swap(idx[k], idx[r]);
        }
        vector<int> chosen(idx.begin(), idx.begin() + m);
        sort(chosen.begin(), chosen.end());

        snprintf(buf, sizeof(buf), "%d", m);
        out += buf;
        for (int k = 0; k < m; k++) {
            int a = rnd.next(10, 100);
            snprintf(buf, sizeof(buf), " %d %d", chosen[k], a);
            out += buf;
        }
        out += '\n';
    }

    fputs(out.c_str(), stdout);
    return 0;
}
