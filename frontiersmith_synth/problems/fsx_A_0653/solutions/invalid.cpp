// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: claims a rope whose position is a huge, obviously
// out-of-range index for lane 1 (no realistic lane has >= 1e9 markers). The
// checker's bounded read on 'pos' must reject this immediately.
int main() {
    long long M, Q, B;
    if (!(cin >> M >> Q >> B)) return 0;
    for (long long c = 1; c <= M; c++) {
        if (c == 1) {
            cout << 1 << "\n";
            cout << 1000000000LL << " " << 1000000001LL << "\n";
        } else {
            cout << 0 << "\n";
        }
    }
    return 0;
}
