// TIER: invalid
// Deliberately infeasible: the first tower's channel is out of range [1,C].
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
        int c = (i == 1) ? (C + 5) : 1; // out-of-range channel on tower 1
        printf("%d%c", c, i == N ? '\n' : ' ');
    }
    return 0;
}
