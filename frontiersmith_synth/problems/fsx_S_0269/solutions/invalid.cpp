// TIER: invalid
// Deliberately infeasible: start every stop at time 0. Since every convoy has L>=2 stops
// and green windows are >=1, stop 2 starts before stop 1 finishes -> precedence violated
// -> checker must score this 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int M, J;
    if (scanf("%d %d", &M, &J) != 2) return 0;
    for (int j = 0; j < J; j++) {
        int L; scanf("%d", &L);
        for (int i = 0; i < L; i++) { int m, p; scanf("%d %d", &m, &p); printf("0 "); }
        printf("\n");
    }
    return 0;
}
