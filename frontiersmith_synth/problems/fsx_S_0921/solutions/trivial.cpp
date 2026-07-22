// TIER: trivial
// Dice the finest possible grid (every legal line drawn, every rectangle a
// single 1x1 cell) and realize the base 1x1 die on every defect-free cell.
// This is EXACTLY the checker's own internal baseline construction, so it
// always scores ratio == 0.100000.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, K;
    cin >> N >> K;
    vector<int> A(K + 1), B(K + 1);
    vector<ll> V(K + 1);
    for (int i = 1; i <= K; i++) cin >> A[i] >> B[i] >> V[i];
    int D;
    cin >> D;
    vector<vector<char>> grid(N + 2, vector<char>(N + 2, 0));
    for (int i = 0; i < D; i++) {
        int r, c;
        cin >> r >> c;
        grid[r][c] = 1;
    }

    int unitIdx = -1;
    for (int i = 1; i <= K; i++)
        if (A[i] == 1 && B[i] == 1) { unitIdx = i; break; }

    cout << (N - 1);
    for (int p = 1; p <= N - 1; p++) cout << ' ' << p;
    cout << '\n';
    cout << (N - 1);
    for (int p = 1; p <= N - 1; p++) cout << ' ' << p;
    cout << '\n';

    for (int r = 1; r <= N; r++) {
        for (int c = 1; c <= N; c++) {
            if (!grid[r][c] && unitIdx != -1) cout << unitIdx << ' ';
            else cout << 0 << ' ';
        }
    }
    cout << '\n';
    return 0;
}
