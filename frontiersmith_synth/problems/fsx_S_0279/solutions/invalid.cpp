// TIER: invalid
// Deliberately infeasible: deploy two type-1 arrays anchored at the same cell so they
// overlap -> the checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, T;
    if (scanf("%d %d %d", &H, &W, &T) != 3) return 0;
    // consume the rest of the input, we don't need it
    printf("2\n");
    printf("1 0 0 0\n");
    printf("1 0 0 0\n");   // identical placement -> guaranteed overlap
    return 0;
}
