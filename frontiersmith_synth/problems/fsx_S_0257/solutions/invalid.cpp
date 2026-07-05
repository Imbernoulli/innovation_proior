// TIER: invalid
// Start every stage at minute 0. With M >= 2 stages per loop this violates precedence
// (stage 2 would start before stage 1 finishes) and overlaps every unit => score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    int mm, dd;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) scanf("%d %d", &mm, &dd);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (j) printf(" ");
            printf("0");
        }
        printf("\n");
    }
    return 0;
}
