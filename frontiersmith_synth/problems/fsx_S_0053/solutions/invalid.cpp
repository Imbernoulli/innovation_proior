// TIER: invalid
// Deliberately infeasible: assigns crew 0 to every phase (often ineligible) and starts
// every phase at time 0 (crew-overlap + eligibility violations). Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int j = 0; j < n; j++) {
        int o;
        scanf("%d", &o);
        for (int k = 0; k < o; k++) {
            int e;
            scanf("%d", &e);
            for (int idx = 0; idx < e; idx++) {
                int c, d;
                scanf("%d %d", &c, &d);
            }
        }
        for (int k = 0; k < o; k++)
            printf("0 0%c", k + 1 == o ? '\n' : ' ');
    }
    return 0;
}
