// TIER: invalid
// Deliberately infeasible: emits label q (out of range [0,q-1]) for every node.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, q, H;
    if (scanf("%d %d %d %d", &n, &m, &q, &H) != 4) return 0;
    // skip the rest of the input (not needed)
    for (int i = 1; i <= n; i++) printf("%d ", q); // q is out of [0,q-1]
    printf("\n");
    return 0;
}
