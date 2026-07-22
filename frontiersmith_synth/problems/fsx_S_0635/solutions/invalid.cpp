// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, B, sr, sc;
    scanf("%d %d %d", &R, &C, &B);
    scanf("%d %d", &sr, &sc);
    vector<string> grid(R);
    for (int i = 0; i < R; i++) { char buf[210]; scanf("%s", buf); grid[i] = buf; }
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) { int x; scanf("%d", &x); }
    // Deliberately infeasible: place a mirror directly on the source cell.
    printf("DR\n1\n%d %d /\n", sr, sc);
    return 0;
}
