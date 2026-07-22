// TIER: invalid
// Deliberately infeasible: claims a firebreak cell far outside the grid.
// The checker's bounded ouf.readInt(0, W-1, "x") rejects this immediately.
#include <cstdio>
int main() {
    printf("1\n");
    printf("0 999999999\n");
    return 0;
}
