#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, depot;
vector<int> X, Y, D;
vector<int> P, Q, W;

static inline ll darr(int a, int b) {
    return (ll)abs(X[a] - X[b]) + abs(Y[a] - Y[b]) + D[b];
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    depot = inf.readInt();

    X.assign(N + 1, 0); Y.assign(N + 1, 0); D.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) {
        X[i] = inf.readInt();
        Y[i] = inf.readInt();
        D[i] = inf.readInt();
    }
    P.assign(M + 1, 0); Q.assign(M + 1, 0); W.assign(M + 1, 0);
    ll B = 0;
    for (int j = 1; j <= M; j++) {
        P[j] = inf.readInt();
        Q[j] = inf.readInt();
        W[j] = inf.readInt();
        B += W[j];
    }
    if (B <= 0) quitf(_fail, "bad instance: total penalty B=%lld", B);

    // ---- read & validate participant's stop-event sequence ----
    int K = ouf.readInt(0, 2 * M, "K");
    vector<char> pickSeen(M + 1, 0), dropSeen(M + 1, 0);
    ll T = 0;
    int prev = depot;
    for (int e = 0; e < K; e++) {
        int type = ouf.readInt(1, 2, "type");
        int j = ouf.readInt(1, M, "job");
        int loc;
        if (type == 1) {
            if (pickSeen[j]) quitf(_wa, "job %d picked up more than once", j);
            pickSeen[j] = 1;
            loc = P[j];
        } else {
            if (dropSeen[j]) quitf(_wa, "job %d delivered more than once", j);
            if (!pickSeen[j]) quitf(_wa, "job %d delivered before it was picked up", j);
            dropSeen[j] = 1;
            loc = Q[j];
        }
        T += darr(prev, loc);
        prev = loc;
    }
    if (K > 0) T += darr(prev, depot); // return to garage
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // every job must be either fully served (pick+drop) or fully unserved
    ll U = 0;
    for (int j = 1; j <= M; j++) {
        if (pickSeen[j] != dropSeen[j])
            quitf(_wa, "job %d picked up but never delivered (or vice versa)", j);
        if (!pickSeen[j]) U += W[j]; // unserved -> forfeit penalty
    }

    ll F = T + U;
    if (F < 0) quitf(_fail, "negative objective F=%lld", F);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld T=%lld U=%lld Ratio: %.6f", F, B, T, U, sc / 1000.0);
    return 0;
}
