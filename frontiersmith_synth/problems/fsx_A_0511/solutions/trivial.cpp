// TIER: trivial
// The naive lexicographic greedy walk = EXACTLY the checker's internal baseline B.
// Start at the smallest vertex with an allowed out-edge, always append the
// smallest legal-and-unused symbol, stop when stuck.  Covers F = B -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int A, K, F;
    scanf("%d %d %d", &A, &K, &F);
    int V = 1; for (int i = 0; i < K - 1; i++) V *= A;
    ll E = (ll)V * A;

    vector<char> forb(E, 0);
    static char buf[64];
    for (int t = 0; t < F; t++) {
        scanf("%s", buf);
        ll code = 0;
        for (int i = 0; i < K; i++) code = code * A + (buf[i] - '0');
        forb[code] = 1;
    }

    int start = -1;
    for (int v = 0; v < V && start < 0; v++)
        for (int s = 0; s < A; s++)
            if (!forb[(ll)v * A + s]) { start = v; break; }
    if (start < 0) { printf("0\n"); return 0; }  // degenerate guard

    // spell the (K-1)-mer of the start vertex
    string S;
    S.reserve(1 << 16);
    {
        vector<int> digs(K - 1, 0);
        int x = start;
        for (int i = K - 2; i >= 0; i--) { digs[i] = x % A; x /= A; }
        for (int i = 0; i < K - 1; i++) S.push_back((char)('0' + digs[i]));
    }

    vector<char> used(E, 0);
    int cur = start;
    while (true) {
        int chosen = -1;
        for (int s = 0; s < A; s++) {
            ll ec = (ll)cur * A + s;
            if (!forb[ec] && !used[ec]) { chosen = s; break; }
        }
        if (chosen < 0) break;
        ll ec = (ll)cur * A + chosen;
        used[ec] = 1;
        S.push_back((char)('0' + chosen));
        cur = (int)(ec % V);
    }

    printf("%s\n", S.c_str());
    return 0;
}
