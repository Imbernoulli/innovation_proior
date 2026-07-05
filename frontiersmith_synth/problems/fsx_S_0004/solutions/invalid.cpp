// TIER: invalid
// Deliberately infeasible: deliver request 1 before ever picking it up.
#include <cstdio>
int main() {
    printf("2\n");
    printf("1 1\n"); // delivery of request 1 ...
    printf("0 1\n"); // ... pickup afterwards -> precedence violation -> score 0
    return 0;
}
