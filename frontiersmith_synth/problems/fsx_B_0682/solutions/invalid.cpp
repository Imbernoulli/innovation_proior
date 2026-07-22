// TIER: invalid
// Deliberately infeasible: (0,0) is always outside the casting cavity (the generator
// always leaves a full '#' margin row/column at the grid border), so this seed is
// rejected by the checker's feasibility check.
#include <bits/stdc++.h>
using namespace std;
int main() {
    printf("1\n0 0 0\n");
    return 0;
}
