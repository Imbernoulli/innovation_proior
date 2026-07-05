// TIER: invalid
// Deliberately infeasible: claims to shut one valve but prints an out-of-range
// edge index (0), which the checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, s, t; long long budget;
    if (!(cin >> n >> m >> s >> t >> budget)) { printf("1\n0\n"); return 0; }
    printf("1\n0\n"); // index 0 is out of the valid range [1, m]
    return 0;
}
