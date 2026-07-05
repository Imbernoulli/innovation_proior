// TIER: trivial
// Do-nothing baseline: shut no valves. Always feasible; scores the calibration point (~0.1).
#include <bits/stdc++.h>
using namespace std;
int main() {
    // Consume the input (not strictly required) then print an empty closure set.
    int n, m, s, t; long long budget;
    if (!(cin >> n >> m >> s >> t >> budget)) { printf("0\n"); return 0; }
    printf("0\n");
    return 0;
}
