// TIER: trivial
// All bees WEST (all zeros) -- exactly the grader's internal baseline, ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    // consume/ignore the routes; we only need n
    for (int v = 1; v <= n; v++) printf("%d%c", 0, v == n ? '\n' : ' ');
    return 0;
}
