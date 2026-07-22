// TIER: invalid
// Deliberately infeasible: the first weight is negative, out of the required
// [0, 1e12] range -> the checker's bounded readDouble rejects it immediately,
// no Ratio is ever printed, score 0.
#include <cstdio>
int main() {
    printf("-1");
    for (int i = 1; i < 72; i++) printf(" 1");
    printf("\n");
    return 0;
}
