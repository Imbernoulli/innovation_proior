// TIER: invalid
// Deliberately infeasible: prints home team 0 (not a competitor, out of [1,n])
// for every game, so the checker must reject it and score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, R; ll lambda;
    if (scanf("%d %d %lld", &n, &R, &lambda) != 3) return 0;
    int G = n / 2;
    for (int r = 0; r < R; r++)
        for (int g = 0; g < G; g++) printf("0%c", g + 1 < G ? ' ' : '\n');
    return 0;
}
