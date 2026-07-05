// TIER: invalid
// Deliberately infeasible: emits an hour far outside every berth window
// (t = 1000000000), so the checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int T, J, G;
    if (scanf("%d %d %d", &T, &J, &G) != 3) return 0;
    for (int t = 0; t < T; t++) { int a,b,c; scanf("%d %d %d", &a,&b,&c); }
    for (int j = 0; j < J; j++) {
        int a, b, w; scanf("%d %d %d", &a, &b, &w);
        for (int k = 0; k < w; k++) printf("1000000000 1 ");
        printf("\n");
    }
    return 0;
}
