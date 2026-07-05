// TIER: invalid
// Deliberately infeasible for every N: it emits N-1 runs but they are all the SAME pair
// "1 2", so after the first the checker sees a duplicate cable run (and the network is also
// disconnected).  Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    for (int i = 0; i < N; i++) {
        int x, y, c, w;
        scanf("%d %d %d %d", &x, &y, &c, &w);
    }
    printf("%d\n", N - 1);
    for (int i = 1; i < N; i++) printf("1 2\n");
    return 0;
}
