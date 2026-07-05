// TIER: trivial
// Put every radio on channel 1 -> exactly the checker's baseline B -> ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K, M;
    if (scanf("%d %d %d", &N, &K, &M) != 3) return 0;
    for (int i = 0; i < M; i++) {
        int u, v, s, p;
        scanf("%d %d %d %d", &u, &v, &s, &p);
    }
    for (int i = 0; i < N; i++) printf("%d%c", 1, i + 1 == N ? '\n' : ' ');
    return 0;
}
