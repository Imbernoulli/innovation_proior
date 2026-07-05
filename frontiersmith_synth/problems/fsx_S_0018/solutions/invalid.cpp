// TIER: invalid
// Deliberately infeasible: stage EVERY act on platform 1, overloading its generator.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, n;
    if (scanf("%d %d", &m, &n) != 2) return 0;
    // consume the rest of the input (not needed, but keep parsing clean)
    // just emit platform 1 for every act -> total draw far exceeds C[1].
    for (int i = 1; i <= n; i++) printf("1%c", i == n ? '\n' : ' ');
    return 0;
}
