// TIER: invalid
// Deliberately infeasible: assign EVERY block to pump 1, overflowing its reservoir.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int P, B;
    if (scanf("%d %d", &P, &B) != 2) return 0;
    vector<long long> cap(P + 1);
    for (int p = 1; p <= P; p++) scanf("%lld", &cap[p]);
    int a, b;
    for (int j = 1; j <= B; j++)
        for (int p = 1; p <= P; p++) { scanf("%d %d", &a, &b); }

    for (int j = 1; j <= B; j++) printf("1%c", j == B ? '\n' : ' ');
    return 0;
}
