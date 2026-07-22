// TIER: invalid
// Deliberately infeasible: claims to place one mirror far outside the grid. Must
// score 0 (checker's bounded reads reject the out-of-range position).
#include <cstdio>
int main() {
    int n, M, ec;
    char group[8];
    scanf("%d %d %7s", &n, &M, group);
    scanf("%d", &ec);
    printf("1\n");
    printf("999999 999999 /\n");
    return 0;
}
