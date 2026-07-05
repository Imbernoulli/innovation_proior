// TIER: invalid
// Deliberately infeasible: names a hub index out of range -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M; long long C;
    if (scanf("%d %d %lld", &N, &M, &C) != 3) return 0;
    // out-of-range hub index (> N) makes the output infeasible
    printf("2\n%d %d\n", 1, N + 1000000);
    return 0;
}
