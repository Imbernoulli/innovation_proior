// TIER: trivial
// Build one ladder at every spawning ground. Always feasible; equals the checker baseline B.
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
    vector<int> dem(D);
    for (int i = 0; i < D; i++) scanf("%d", &dem[i]);
    printf("%d\n", D);
    for (int i = 0; i < D; i++) printf("%d%c", dem[i], i + 1 == D ? '\n' : ' ');
    if (D == 0) printf("\n");
    return 0;
}
