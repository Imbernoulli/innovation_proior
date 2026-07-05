// TIER: invalid
// Deliberately infeasible: feature ALL performers. Since every test has at least one
// conflict edge, this roster contains a conflicting pair and must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);
    // We don't even need the edges; featuring everyone violates independence.
    printf("%d\n", n);
    for (int i = 1; i <= n; i++) printf("%d\n", i);
    return 0;
}
