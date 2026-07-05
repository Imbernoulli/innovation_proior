// TIER: invalid
// Deliberately infeasible: build no ladders while there is at least one spawning ground.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M;
    scanf("%d %d", &N, &M);
    for (int i = 0; i < M; i++) { int u, v; scanf("%d %d", &u, &v); }
    vector<int> c(N + 1), r(N + 1);
    for (int i = 1; i <= N; i++) scanf("%d", &c[i]);
    for (int i = 1; i <= N; i++) scanf("%d", &r[i]);
    int D; scanf("%d", &D);
    for (int i = 0; i < D; i++) { int x; scanf("%d", &x); }
    printf("0\n");   // no ladders -> spawning grounds uncovered -> infeasible
    return 0;
}
