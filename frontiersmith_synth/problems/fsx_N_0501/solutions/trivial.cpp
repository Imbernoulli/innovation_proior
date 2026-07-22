// TIER: trivial
// Select the guaranteed fallback tree, exactly matching the checker's baseline B.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, T, P;
    if (!(cin >> N >> M >> T >> P)) return 0;
    int O, R;
    cin >> O >> R;
    for (int i = 0; i < N; i++) {
        int w;
        cin >> w;
    }
    for (int t = 0; t < T; t++) {
        int h;
        cin >> h;
    }
    for (int i = 0; i < M; i++) {
        int u, v, c, l, r, a, b;
        cin >> u >> v >> c >> l >> r >> a >> b;
    }
    cout << N - 1 << "\n";
    for (int i = 1; i <= N - 1; i++) cout << i << "\n";
    return 0;
}
