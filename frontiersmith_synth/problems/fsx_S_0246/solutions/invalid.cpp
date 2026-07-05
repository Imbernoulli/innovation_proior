// TIER: invalid
// Deliberately infeasible: assign the same group twice (violates distinctness).
// The checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int V, G;
    if (scanf("%d %d", &V, &G) != 2) return 0;
    vector<int> C(G + 1);
    for (int j = 1; j <= G; j++) scanf("%d", &C[j]);
    for (int i = 1; i <= V; i++)
        for (int j = 1; j <= G; j++) { int a, p; scanf("%d %d", &a, &p); }
    // group 1 assigned to two galleries -> not distinct -> infeasible
    if (G >= 2) printf("2\n1 1\n1 2\n");
    else printf("2\n1 1\n1 1\n");
    return 0;
}
