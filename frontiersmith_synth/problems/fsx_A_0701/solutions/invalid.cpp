// TIER: invalid
// Deliberately infeasible: two vertices share the same (class,sheet) slot.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    scanf("%d %d", &N, &M);
    for (int i = 0; i < M; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
    }
    printf("%d %d\n", 2, N);
    for (int i = 0; i < N; i++) {
        // every vertex claims slot (0,0) -- violates the distinctness feasibility rule.
        printf("%d %d\n", 0, 0);
    }
    printf("%d\n", 0);
    return 0;
}
