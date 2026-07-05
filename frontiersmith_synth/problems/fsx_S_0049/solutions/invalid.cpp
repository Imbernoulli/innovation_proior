// TIER: invalid
// Deliberately infeasible: prints an out-of-range road index (0), which the checker
// rejects via bounded reads -> score 0.
#include <cstdio>
int main() {
    printf("1\n");
    printf("0\n"); // index 0 is out of [1..m] -> infeasible
    return 0;
}
