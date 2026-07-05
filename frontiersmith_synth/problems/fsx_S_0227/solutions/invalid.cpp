// TIER: invalid
// Deliberately infeasible: grant BOTH endpoints of the first listed conflict pair, which
// violates the conflict-free constraint and must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N; long long M;
    if (scanf("%d %lld", &N, &M) != 2) return 0;
    for (int i = 1; i <= N; i++) { long long w; scanf("%lld", &w); }
    int a = 1, b = 2;
    if (M >= 1) { scanf("%d %d", &a, &b); }
    for (long long i = 1; i < M; i++) { int x, y; scanf("%d %d", &x, &y); }
    if (a == b) b = (a == 1 ? 2 : 1);
    printf("2\n%d %d\n", a, b);
    return 0;
}
