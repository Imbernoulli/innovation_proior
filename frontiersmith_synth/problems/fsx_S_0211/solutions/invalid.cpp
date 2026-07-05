// TIER: invalid
// Deliberately infeasible: prints a single self-loop link (a == b), which the checker
// rejects, so this must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K;
    if (scanf("%d %d", &N, &K) != 2) return 0;
    for (int i = 0; i < N; i++) {
        int x, y, c;
        scanf("%d %d %d", &x, &y, &c);
    }
    printf("1\n");
    printf("1 1\n");
    return 0;
}
