// TIER: invalid
// Deliberately infeasible: builds a single guaranteed link, leaving most nodes disconnected.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    for (int i = 1; i <= N; i++) { int c; scanf("%d", &c); }
    // (1,2) is a guaranteed buildable link, but this cannot connect N>=3 nodes.
    printf("1\n1 2\n");
    return 0;
}
