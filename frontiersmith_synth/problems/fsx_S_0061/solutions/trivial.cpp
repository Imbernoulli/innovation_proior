// TIER: trivial
// Do-nothing baseline: cut no fiber. Feasible by definition, scores the calibration point
// (Ratio == 0.1, since F == B0).
#include <bits/stdc++.h>
using namespace std;
int main() {
    // We do not even need to read the graph; cutting nothing is always feasible.
    printf("0\n");
    return 0;
}
