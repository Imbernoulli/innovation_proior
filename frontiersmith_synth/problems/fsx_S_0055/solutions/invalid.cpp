// TIER: invalid
// Deliberately infeasible: prints a single self-loop lift (a == b), which the checker
// rejects, so this must score 0 on every test.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    for (int i = 0; i < N; i++) {
        int x, y, h, c;
        scanf("%d %d %d %d", &x, &y, &h, &c);
    }
    printf("1\n");
    printf("1 1\n");   // self-loop -> infeasible
    return 0;
}
