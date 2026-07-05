// TIER: invalid
// Deliberately infeasible: prints out-of-range mode values (5) instead of 0/1,
// so the checker's bounded read rejects it and it scores 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 0; i < n; i++) printf("5%c", i + 1 == n ? '\n' : ' ');
    return 0;
}
