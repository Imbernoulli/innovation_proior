// TIER: greedy
// The obvious single-pass recipe: find the single highest-value catalog die
// (value grows faster than area, so this is always the biggest square) and
// dice the WHOLE wafer at that die's fixed pitch, starting from the corner,
// completely ignoring where the defects are. Every resulting cell that
// contains a defective cell is simply wasted -- because the cut lines are
// global and fixed to one pitch, a dense defect band anywhere on the wafer
// poisons every uniform cell it touches, everywhere along that band.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, K;
vector<int> A, Bd;
vector<ll> V;
vector<vector<char>> grid;
vector<vector<ll>> pre;

ll rectDefects(int r1, int r2, int c1, int c2) {
    return pre[r2][c2] - pre[r1 - 1][c2] - pre[r2][c1 - 1] + pre[r1 - 1][c1 - 1];
}
bool fits(int idx, ll w, ll h) {
    ll a = A[idx], b = Bd[idx];
    return (a <= w && b <= h) || (b <= w && a <= h);
}

int main() {
    cin >> N >> K;
    A.assign(K + 1, 0);
    Bd.assign(K + 1, 0);
    V.assign(K + 1, 0);
    for (int i = 1; i <= K; i++) cin >> A[i] >> Bd[i] >> V[i];
    int D;
    cin >> D;
    grid.assign(N + 2, vector<char>(N + 2, 0));
    for (int i = 0; i < D; i++) {
        int r, c;
        cin >> r >> c;
        grid[r][c] = 1;
    }
    pre.assign(N + 1, vector<ll>(N + 1, 0));
    for (int r = 1; r <= N; r++)
        for (int c = 1; c <= N; c++)
            pre[r][c] = grid[r][c] + pre[r - 1][c] + pre[r][c - 1] - pre[r - 1][c - 1];

    int best = 1;
    for (int i = 2; i <= K; i++) if (V[i] > V[best]) best = i;
    int pr = A[best], pc = Bd[best];

    vector<int> hcut, vcut;
    for (int p = pr; p < N; p += pr) hcut.push_back(p);
    for (int p = pc; p < N; p += pc) vcut.push_back(p);

    vector<pair<int, int>> rowBands, colBands;
    { int prev = 1; for (int p : hcut) { rowBands.push_back({prev, p}); prev = p + 1; } rowBands.push_back({prev, N}); }
    { int prev = 1; for (int p : vcut) { colBands.push_back({prev, p}); prev = p + 1; } colBands.push_back({prev, N}); }

    cout << hcut.size(); for (int p : hcut) cout << ' ' << p; cout << '\n';
    cout << vcut.size(); for (int p : vcut) cout << ' ' << p; cout << '\n';

    for (auto& rb : rowBands) {
        int h = rb.second - rb.first + 1;
        for (auto& cb : colBands) {
            int w = cb.second - cb.first + 1;
            ll defc = rectDefects(rb.first, rb.second, cb.first, cb.second);
            if (defc > 0) { cout << 0 << ' '; continue; }
            int bestIdx = 0; ll bestVal = 0;
            for (int i = 1; i <= K; i++)
                if (fits(i, w, h) && V[i] > bestVal) { bestVal = V[i]; bestIdx = i; }
            cout << bestIdx << ' ';
        }
    }
    cout << '\n';
    return 0;
}
