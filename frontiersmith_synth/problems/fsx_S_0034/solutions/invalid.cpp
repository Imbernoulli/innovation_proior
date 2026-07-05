// TIER: invalid
// Deliberately infeasible: prints a ride setting of 2 (out of {0,1}) -> must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int v = 0; v < n; v++) printf("%s2", v ? " " : "");
    printf("\n");
    return 0;
}
