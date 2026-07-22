// TIER: invalid
// Deliberately infeasible: crew 1 claims to move along street id 999999,
// which is always out of range [1,M] -> the checker's bounded read rejects
// it -> no Ratio emitted -> score 0. Remaining crews print 0 moves.
#include <cstdio>
int main() {
    int N, M, K;
    scanf("%d %d %d", &N, &M, &K);
    printf("1\n999999 P 0\n");
    for (int p = 2; p <= K; p++) printf("0\n");
    return 0;
}
