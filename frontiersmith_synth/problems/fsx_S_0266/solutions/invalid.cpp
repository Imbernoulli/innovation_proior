// TIER: invalid
// Build zero relays -> nothing is illuminated -> infeasible -> must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, r;
    scanf("%d %d %d", &N, &M, &r);
    for (int i = 0; i < M; i++) { int u, v; scanf("%d %d", &u, &v); }
    for (int i = 0; i < N; i++) { int c; scanf("%d", &c); }
    printf("0\n");
    return 0;
}
