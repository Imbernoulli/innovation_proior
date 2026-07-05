// TIER: invalid
// Deliberately infeasible: installs a single link, leaving everyone else isolated
// -> the backbone is not connected -> must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    for (int i = 0; i < n; i++) { long long x, y; int b; scanf("%lld %lld %d", &x, &y, &b); }
    printf("1\n1 2\n");
    return 0;
}
