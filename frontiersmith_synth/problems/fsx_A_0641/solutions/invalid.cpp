// TIER: invalid
// Deliberately infeasible: repeats value 0 twice and never outputs p-1, so it is not a
// permutation. Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int p;
    if (scanf("%d", &p) != 1) return 0;
    for (int d = 1; d <= p - 1; d++) { int w; scanf("%d", &w); }
    for (int i = 0; i < p; i++) {
        int v = (i == p - 1) ? 0 : i; // duplicate 0, missing p-1
        printf("%d%c", v, i + 1 == p ? '\n' : ' ');
    }
    return 0;
}
