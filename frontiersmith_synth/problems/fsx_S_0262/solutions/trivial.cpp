// TIER: trivial
// All stations low-power (all-zero) = the dormant baseline the checker measures.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    // consume the rest of the input (not needed for the baseline plan)
    // just print n zeros.
    for (int i = 0; i < n; i++) printf("0%c", i + 1 == n ? '\n' : ' ');
    return 0;
}
