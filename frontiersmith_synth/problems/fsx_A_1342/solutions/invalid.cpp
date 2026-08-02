// TIER: invalid
// Deliberately infeasible: reuses cell 0 in two different couples. Must score 0.
#include <cstdio>
int main() {
    printf("2\n0 1\n0 2\n");
    return 0;
}
