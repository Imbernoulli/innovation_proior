// TIER: invalid
// Deliberately infeasible: start every operation at time 0. Whenever a module
// has two or more operations, the second one starts before the first finishes
// (precedence violation), and any two operations sharing a workstation overlap.
// The checker must reject this -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    long long cnt = 0;
    for (int j = 0; j < J; j++) {
        int k; scanf("%d", &k);
        for (int o = 0; o < k; o++) {
            int m, d; scanf("%d %d", &m, &d);
            cnt++;
        }
    }
    for (long long i = 0; i < cnt; i++)
        printf("0\n");
    return 0;
}
