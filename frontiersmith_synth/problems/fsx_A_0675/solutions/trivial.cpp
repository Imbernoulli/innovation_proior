// TIER: trivial
// Address-modulo / round-robin assignment: c_i = ((i-1) mod S) + 1.
// This is exactly the checker's internal baseline construction.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int P, S, K, T;
    scanf("%d %d %d %d", &P, &S, &K, &T);
    for (int t = 0; t < T; t++) { int x; scanf("%d", &x); }

    for (int i = 1; i <= P; i++) {
        printf("%d%c", ((i - 1) % S) + 1, (i < P) ? ' ' : '\n');
    }
    return 0;
}
