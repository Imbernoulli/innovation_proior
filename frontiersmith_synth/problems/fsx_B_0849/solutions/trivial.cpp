// TIER: trivial
// The checker's own baseline: round-robin channel assignment, ignoring every
// weight entirely. Reproduces B exactly, so this scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, C, M;
    scanf("%d %d %d", &N, &C, &M);
    for (int i = 0; i < M; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        for (int s = 0; s < 7; s++) { int w; scanf("%d", &w); }
    }
    for (int i = 1; i <= N; i++) {
        printf("%d%c", ((i - 1) % C) + 1, i == N ? '\n' : ' ');
    }
    return 0;
}
