// TIER: invalid
// Deliberately INFEASIBLE: builds a single tower on cell 1. For any instance with more than
// a handful of cells this leaves the vast majority uncovered, so the checker rejects it and
// it scores 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, R;
    scanf("%d %d %d", &N, &M, &R);
    vector<int> c(N + 1);
    for (int v = 1; v <= N; v++) scanf("%d", &c[v]);
    for (int i = 0; i < M; i++) { int a, b; scanf("%d %d", &a, &b); }
    printf("1\n1\n");
    return 0;
}
