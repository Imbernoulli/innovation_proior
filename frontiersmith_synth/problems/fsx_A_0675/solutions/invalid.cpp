// TIER: invalid
// Prints an out-of-range room id for the last actor (S+5, outside [1,S]) so the checker
// must reject it as infeasible.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int P, S, K, T;
    scanf("%d %d %d %d", &P, &S, &K, &T);
    for (int t = 0; t < T; t++) { int x; scanf("%d", &x); }

    for (int i = 1; i <= P; i++) {
        int val = ((i - 1) % S) + 1;
        if (i == P) val = S + 5; // deliberately infeasible
        printf("%d%c", val, (i < P) ? ' ' : '\n');
    }
    return 0;
}
