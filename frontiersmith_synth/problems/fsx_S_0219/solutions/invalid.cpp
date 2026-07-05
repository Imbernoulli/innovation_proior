// TIER: invalid
// Deliberately infeasible: places the 2x2 square twice at the same anchor (0,0), which the
// generator guarantees is free. The two beds overlap, so the checker must score this 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, D;
    scanf("%d %d %d", &H, &W, &D);
    // (we don't even need the rest of the input to emit an overlapping placement)
    printf("2\n");
    printf("0 0 0 0\n");
    printf("0 0 0 0\n");
    return 0;
}
