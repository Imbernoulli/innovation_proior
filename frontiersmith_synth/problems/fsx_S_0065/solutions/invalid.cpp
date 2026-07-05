// TIER: invalid
// Deliberately infeasible: start every operation at time 0. For any order with
// M>=2 steps this violates route precedence (step 1 would start before step 0
// finishes), so the checker rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    for (int j = 0; j < J; j++)
        for (int o = 0; o < M; o++) { int b; ll d; scanf("%d %lld", &b, &d); }
    for (int j = 0; j < J; j++) {
        for (int o = 0; o < M; o++) {
            printf("0");
            if (o + 1 < M) printf(" ");
        }
        printf("\n");
    }
    return 0;
}
