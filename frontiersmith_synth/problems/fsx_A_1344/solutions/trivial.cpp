// TIER: trivial
// Do nothing at all: never spend the blocking budget. This is the checker's
// calibration floor (weaker than its own internal "seal the nearest gate"
// baseline, which is what Ratio is measured against).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, T, S, K; long long B;
    if (!(cin >> N >> T >> S >> K >> B)) return 0;
    int rx, ry; cin >> rx >> ry;
    for (int r = 0; r < N; r++) { string row; cin >> row; }
    for (int t = 0; t < T; t++) cout << 0 << "\n";
    return 0;
}
