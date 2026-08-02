// TIER: trivial
// Always remove exactly 1 stone from the lowest-indexed non-empty cairn.
// Ignores fracture limits and all game theory -- this is literally the
// checker's own internal baseline, so it always scores ratio ~= 0.10.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T;
    if (scanf("%d", &T) != 1) return 0;
    for (int k = 0; k < T; k++) {
        int M; scanf("%d", &M);
        vector<int> R(M), N(M);
        for (int i = 0; i < M; i++) scanf("%d %d", &R[i], &N[i]);
        int idx = -1;
        for (int i = 0; i < M; i++) if (N[i] >= 1) { idx = i; break; }
        printf("%d %d\n", idx, N[idx] - 1);
    }
    return 0;
}
