// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: eliminates vertex 0 "guarded" by itself, which the checker
// must reject outright (a guard set can never legally include the vertex being
// eliminated -- it covers nothing new and the escape route through any other live
// neighbor is left completely unwatched).
int main() {
    long long n, m;
    if (!(cin >> n >> m)) return 0;
    cout << "0 1 0\n";
    return 0;
}
