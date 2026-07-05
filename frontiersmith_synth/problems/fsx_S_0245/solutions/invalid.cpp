// TIER: invalid
// Starts every stage at time 0. Inverters with >1 stage violate precedence and bays get
// double-booked, so the schedule is infeasible and must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    for (int i = 0; i < J; i++) {
        int L; scanf("%d", &L);
        for (int k = 0; k < L; k++) { int m, d; scanf("%d %d", &m, &d); }
        for (int k = 0; k < L; k++) printf("0%c", k + 1 == L ? '\n' : ' ');
    }
    return 0;
}
