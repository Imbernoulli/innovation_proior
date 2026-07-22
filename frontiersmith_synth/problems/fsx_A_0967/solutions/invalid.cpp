// TIER: invalid
// Deliberately infeasible: a single tile covering the whole tapestry, but with a compression
// level equal to K (one past the maximum valid level K-1). The checker's bounded read of the
// level token must reject this -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, K;
    if (scanf("%d %d %d", &R, &C, &K) != 3) return 0;
    printf("1\n0 %d\n1\n0 %d\n%d\n", R, C, K); // level = K is out of [0, K-1]
    return 0;
}
