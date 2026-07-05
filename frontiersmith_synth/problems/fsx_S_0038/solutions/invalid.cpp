// TIER: invalid
// Deliberately infeasible: build zero watchtowers, leaving every cell
// unmonitored (N >= 3). The checker must reject this -> score 0.
#include <cstdio>
int main() {
    printf("0\n");
    return 0;
}
