// TIER: invalid
// Deliberately infeasible: emit channel 0, which is out of the 1..K range -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K, M;
    if (scanf("%d %d %d", &N, &K, &M) != 3) return 0;
    for (int i = 0; i < M; i++) {
        int u, v, s, p;
        scanf("%d %d %d %d", &u, &v, &s, &p);
    }
    for (int i = 0; i < N; i++) printf("%d%c", 0, i + 1 == N ? '\n' : ' ');
    return 0;
}
