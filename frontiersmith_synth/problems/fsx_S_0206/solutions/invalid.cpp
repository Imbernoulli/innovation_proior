// TIER: invalid
// Deliberately infeasible: builds a single depot at an out-of-range district
// index, which the checker must reject (score 0).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, D, r;
    if (scanf("%d %d %d %d", &n, &m, &D, &r) != 4) return 0;
    // We don't even need the rest; emit an out-of-range depot index.
    printf("1\n");
    printf("%d\n", n + 1); // out of [1..n] -> checker rejects
    return 0;
}
