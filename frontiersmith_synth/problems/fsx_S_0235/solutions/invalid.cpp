// TIER: invalid
// Deliberately infeasible: claim to dig one trench but reference an out-of-range
// index (0), which the checker must reject -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    printf("1\n0\n");
    return 0;
}
