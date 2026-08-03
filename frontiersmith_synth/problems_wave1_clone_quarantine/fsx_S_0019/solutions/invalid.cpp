// TIER: invalid
// Deliberately infeasible: emits a single self-loop trench (a == b), which the checker
// rejects outright, so this must score 0 on every test.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, T;
    if (scanf("%d %d", &N, &T) != 2) return 0;
    for (int i = 0; i < N; i++) { int x, y, c; scanf("%d %d %d", &x, &y, &c); }
    printf("1\n1 1\n");
    return 0;
}
