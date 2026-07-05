// TIER: invalid
// Deliberately infeasible: delivers job 1 before ever picking it up -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, depot;
    if (scanf("%d %d %d", &N, &M, &depot) != 3) return 0;
    for (int i = 0; i < N; i++) { int x, y, d; scanf("%d %d %d", &x, &y, &d); }
    for (int j = 0; j < M; j++) { int p, q, w; scanf("%d %d %d", &p, &q, &w); }
    printf("1\n2 1\n"); // delivery event for job 1 with no preceding pickup
    return 0;
}
