// TIER: invalid
// Deliberately infeasible: claims to issue a step with id 0, which is out of
// the valid range [1, n] -- the checker's bounded read rejects it immediately.
#include <cstdio>
int main() {
    printf("1\n");
    printf("1 0 0 0\n");
    return 0;
}
