// TIER: invalid
// Deliberately infeasible: claims to ADD stones to cairn 0 instead of
// removing any (new count > old count), which the checker must reject.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T;
    if (scanf("%d", &T) != 1) return 0;
    for (int k = 0; k < T; k++) {
        int M; scanf("%d", &M);
        vector<int> R(M), N(M);
        for (int i = 0; i < M; i++) scanf("%d %d", &R[i], &N[i]);
        printf("0 %d\n", N[0] + 5);  // increasing a pile is never legal
    }
    return 0;
}
