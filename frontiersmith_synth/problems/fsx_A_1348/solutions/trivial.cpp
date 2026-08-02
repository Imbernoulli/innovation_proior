// TIER: trivial
// Mirrors the checker's own baseline exactly: cut paths 1..K-1 by input index, clear nothing.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, K, LAMBDA;
    scanf("%d %d %d", &n, &K, &LAMBDA);
    for (int i = 0; i < n; i++) { int t; scanf("%d", &t); }
    for (int i = 0; i < n; i++) { int t; scanf("%d", &t); }
    for (int i = 0; i < n - 1; i++) { int a, b, c; scanf("%d %d %d", &a, &b, &c); }

    printf("0\n");
    printf("%d\n", K - 1);
    for (int i = 1; i <= K - 1; i++) printf("%d ", i);
    printf("\n");
    return 0;
}
