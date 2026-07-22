// TIER: invalid
// Deliberately infeasible: claims a pipe index one past the end of the input list.
#include <cstdio>
int main() {
    int V, E, K;
    scanf("%d %d %d", &V, &E, &K);
    printf("1\n%d\n", E + 1);
    return 0;
}
