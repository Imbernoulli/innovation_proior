// TIER: greedy
// The obvious textbook move: "remove 1..C from a pile" (classic Bounded Nim) has
// Grundy value a mod (C+1) -- a clean fact many coders remember. This solution
// honestly simulates the REAL recurrence (not the formula) up to a = max(S_i), so
// its table is exactly correct there, then assumes -- without ever checking -- that
// max(S_i)+1 is the period and tiles it forever. That assumption is only true when
// S_i is the full interval {1..C}; for a general/sparse move menu it silently goes
// wrong past the window it actually computed, which is most of the query space.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    scanf("%d", &N);
    vector<vector<int>> S(N);
    for (int i = 0; i < N; i++) {
        int K; scanf("%d", &K);
        S[i].resize(K);
        for (int j = 0; j < K; j++) scanf("%d", &S[i][j]);
    }
    for (int i = 0; i < N; i++) {
        int maxS = 0;
        for (int v : S[i]) maxS = max(maxS, v);
        int M = maxS + 1;               // assumed period, unverified
        vector<int> g(M, 0);
        for (int a = 1; a < M; a++) {
            bool seen[16] = {false};
            for (int s : S[i]) if (s <= a) { int v = g[a - s]; if (v < 16) seen[v] = true; }
            int m = 0; while (m < 16 && seen[m]) m++;
            g[a] = m;
        }
        printf("%d", M);
        for (int r = 0; r < M; r++) printf(" %d", g[r]);
        printf("\n");
    }
    return 0;
}
