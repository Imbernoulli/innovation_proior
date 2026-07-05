// TIER: invalid
// Deliberately infeasible: shuts the reservoir s itself, which the checker forbids.
// Must score 0 on every test.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, s, t; long long K;
    if (scanf("%d %d %d %d %lld", &n, &m, &s, &t, &K) != 5) return 0;
    // We only need s to emit an illegal shutdown of the source terminal.
    printf("1\n%d\n", s);
    return 0;
}
