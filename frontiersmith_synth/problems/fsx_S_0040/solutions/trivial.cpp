// TIER: trivial
// Do-nothing baseline: serve no job, van stays at the depot.
// F == sum of all penalties == B  ->  ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, depot;
    if (scanf("%d %d %d", &N, &M, &depot) != 3) return 0;
    for (int i = 0; i < N; i++) { int x, y, d; scanf("%d %d %d", &x, &y, &d); }
    for (int j = 0; j < M; j++) { int p, q, w; scanf("%d %d %d", &p, &q, &w); }
    printf("0\n");
    return 0;
}
