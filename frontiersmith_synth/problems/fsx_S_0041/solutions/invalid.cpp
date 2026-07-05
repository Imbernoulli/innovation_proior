// TIER: invalid
// Deliberately infeasible: assigns asset 0 (out of the valid [1,M] range) to
// every task, so the checker's bounded read rejects it and the score is 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    for (int j = 0; j < J; j++) {
        int k; scanf("%d", &k);
        for (int t = 0; t < k; t++) {
            int e; scanf("%d", &e);
            for (int r = 0; r < e; r++) { int a, d; scanf("%d %d", &a, &d); }
            printf("0 0\n");   // asset 0 is invalid
        }
    }
    return 0;
}
