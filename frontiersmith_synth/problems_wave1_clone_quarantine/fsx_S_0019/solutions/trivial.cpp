// TIER: trivial
// Chamber chain baseline: link artifact chambers 1-2-...-T in input order.
// This is exactly the baseline B the checker measures, so ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, T;
    if (scanf("%d %d", &N, &T) != 2) return 0;
    for (int i = 0; i < N; i++) { int x, y, c; scanf("%d %d %d", &x, &y, &c); }
    printf("%d\n", T - 1);
    for (int i = 1; i < T; i++) printf("%d %d\n", i, i + 1);
    return 0;
}
