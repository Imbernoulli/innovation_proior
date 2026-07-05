// TIER: invalid
// Deliberately infeasible: starts everything at 0 -> every station has overlapping ops.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, F;
    if (scanf("%d %d %d", &n, &m, &F) != 3) return 0;
    for (int a = 0; a < F; a++)
        for (int b = 0; b < F; b++) { long long x; scanf("%lld", &x); }
    for (int j = 0; j < n; j++) {
        int x;
        for (int k = 0; k < 3 * m; k++) scanf("%d", &x);
    }
    string out; out.reserve((size_t)n * m * 2);
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++) {
            out.push_back('0');
            out.push_back(k + 1 < m ? ' ' : '\n');
        }
    fputs(out.c_str(), stdout);
    return 0;
}
