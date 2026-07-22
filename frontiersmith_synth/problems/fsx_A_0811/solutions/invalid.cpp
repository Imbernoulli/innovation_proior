// TIER: invalid
// Deliberately infeasible: claims more valves than the budget allows (k = V+1)
// and lists an out-of-range pipe index among them -- must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, V, C;
    scanf("%d %d %d %d", &N, &M, &V, &C);
    for (int i = 0; i < C; i++) { int c; scanf("%d", &c); }
    printf("%d\n", V + 1);
    for (int i = 0; i < V; i++) printf("1 ");
    printf("%d\n", M + 1000000); // out-of-range pipe index too
    return 0;
}
