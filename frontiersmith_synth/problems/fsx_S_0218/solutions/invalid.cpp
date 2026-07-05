// TIER: invalid
// Deliberately infeasible: install zero regulators, leaving every zone uncovered.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, r;
    scanf("%d %d %d", &n, &m, &r);
    // read and ignore the rest; print an infeasible (empty) cover
    printf("0\n");
    return 0;
}
