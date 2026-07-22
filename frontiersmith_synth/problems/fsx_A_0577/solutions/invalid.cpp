// TIER: invalid
// Deliberately infeasible: print an out-of-range tile index. The checker's bounded
// read ouf.readInt(0, T-1) rejects it -> no Ratio -> scores 0.
#include <cstdio>
int main(){
    int R, C, T;
    if (scanf("%d %d %d", &R, &C, &T) != 3) return 0;
    // one out-of-range index is enough to fail feasibility.
    printf("%d\n", T + 999999);
    return 0;
}
