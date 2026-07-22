// TIER: invalid
// Deliberately infeasible: repeats guest 1 and omits guest N -> checker must
// reject (score 0).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K, M;
    if (!(cin >> N >> K >> M)) return 0;
    for (int i = 0; i < N; i++) { int p; cin >> p; }
    for (int i = 0; i < M; i++) { int u, v, c; cin >> u >> v >> c; }
    for (int g = 1; g <= N; g++) {
        int out = (g == N) ? 1 : g;   // guest N replaced by a duplicate of guest 1
        cout << out << (g < N ? ' ' : '\n');
    }
    return 0;
}
