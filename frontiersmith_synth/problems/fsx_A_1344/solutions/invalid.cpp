// TIER: invalid
// Deliberately infeasible: blocks the runner's own starting cell on turn 1,
// which the checker must reject (score 0).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, T, S, K; long long B;
    cin >> N >> T >> S >> K >> B;
    int rx, ry; cin >> rx >> ry;
    for (int r = 0; r < N; r++) { string row; cin >> row; }
    cout << 1 << " " << rx << " " << ry << "\n";
    for (int t = 1; t < T; t++) cout << 0 << "\n";
    return 0;
}
