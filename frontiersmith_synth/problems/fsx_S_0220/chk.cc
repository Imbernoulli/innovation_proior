#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static vector<long long> X, Y;

static inline long long D(int a, int b) {
    long long dx = X[a] - X[b], dy = Y[a] - Y[b];
    return (long long)llround(sqrt((double)(dx * dx + dy * dy)));
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int N = 2 * n;                       // non-depot nodes 1..2n
    X.assign(N + 1, 0);
    Y.assign(N + 1, 0);
    vector<long long> pen(n + 1, 0);

    X[0] = inf.readInt();                // depot
    Y[0] = inf.readInt();
    for (int i = 1; i <= n; i++) {
        long long px = inf.readInt(), py = inf.readInt();
        long long qx = inf.readInt(), qy = inf.readInt();
        long long P  = inf.readInt();
        X[i] = px;     Y[i] = py;        // pickup i
        X[n + i] = qx; Y[n + i] = qy;    // delivery i
        pen[i] = P;
    }

    // Internal baseline B: serial serve-all tour
    // depot -> p1 -> d1 -> p2 -> d2 -> ... -> pn -> dn -> depot (U = 0).
    long long B = 0;
    {
        int prev = 0;
        for (int i = 1; i <= n; i++) {
            B += D(prev, i);
            B += D(i, n + i);
            prev = n + i;
        }
        B += D(prev, 0);
    }
    if (B <= 0) B = 1;

    // Read participant tour.
    int k = ouf.readInt(0, N, "k");
    vector<int> seq(k);
    vector<char> seen(N + 1, 0);
    vector<int> pos(N + 1, -1);
    for (int r = 0; r < k; r++) {
        int v = ouf.readInt(1, N, format("v[%d]", r));
        if (seen[v]) quitf(_wa, "node %d visited more than once", v);
        seen[v] = 1;
        pos[v] = r;
        seq[r] = v;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // Precedence: any visited delivery must have its pickup visited earlier.
    for (int i = 1; i <= n; i++) {
        if (seen[n + i]) {
            if (!seen[i])
                quitf(_wa, "delivery of relay %d visited but its pickup %d was not", i, i);
            if (pos[i] >= pos[n + i])
                quitf(_wa, "precedence violated: pickup %d at position %d not before delivery %d at position %d",
                      i, pos[i], n + i, pos[n + i]);
        }
    }

    // Objective F = travel distance + skip penalties.
    long long F = 0;
    {
        int prev = 0;
        for (int r = 0; r < k; r++) { F += D(prev, seq[r]); prev = seq[r]; }
        F += D(prev, 0);
    }
    for (int i = 1; i <= n; i++) {
        bool served = seen[i] && seen[n + i];
        if (!served) F += pen[i];
    }
    if (F <= 0) F = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
