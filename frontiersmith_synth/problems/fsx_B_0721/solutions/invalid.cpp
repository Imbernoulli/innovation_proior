// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: two 1x1 mats claiming the SAME cell (violates
// "each free cell covered at most once" and also "exactly one S mat").
// Must score 0.

int main() {
    int H, W;
    scanf("%d %d", &H, &W);
    printf("2\n");
    printf("S 0 0\n");
    printf("S 0 0\n");
    return 0;
}
