// TIER: strong
// The insight: because the cut lines are GLOBAL, the row-cut layout and the
// column-cut layout are two separate 1D spacing decisions. First, a
// row/column whose defect DENSITY is high (a real scarred band) is
// immediately isolated as its own thin, sacrificial band -- quarantining it
// instead of letting it poison an entire uniform big-die cell the way a
// fixed pitch does. Second -- unlike greedy, which always commits to the
// single highest-VALUE catalog die as its uniform pitch regardless of how
// likely a block that size is to actually come up defect-free -- every
// candidate square die's side is TRIED as the clean-region pitch (cheap:
// there are only a few catalog sizes) and the realized total value of each
// full trial dicing is measured directly against the true defect map; the
// candidate that actually scores highest is kept. This is why strong does
// not blindly reach for the biggest die the way greedy does: on a wafer
// with diffuse background noise, a smaller die that reliably survives can
// out-earn a huge die that (almost) never does.
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

// Scan 1..n; a "dirty" position becomes its own singleton band; a run of
// "clean" positions is cut off (band closed) once it reaches bandCap.
vector<int> buildCuts(int n, const vector<char>& dirty, int bandCap) {
    vector<int> cuts;
    int cur = 0;
    for (int r = 1; r <= n; r++) {
        if (dirty[r]) {
            if (cur > 0) { cuts.push_back(r - 1); cur = 0; }
            if (r < n) cuts.push_back(r);
        } else {
            cur++;
            if (cur == bandCap || r == n) {
                if (r < n) cuts.push_back(r);
                cur = 0;
            }
        }
    }
    return cuts;
}

vector<pair<int, int>> bandsFromCuts(const vector<int>& cuts, int n) {
    vector<pair<int, int>> bands;
    int prev = 1;
    for (int p : cuts) { bands.push_back({prev, p}); prev = p + 1; }
    bands.push_back({prev, n});
    return bands;
}

// Assign the best-fitting, defect-free die to every rectangle of a grid;
// returns the total realized value (does not print).
ll evalAssignment(const vector<pair<int, int>>& rowBands, const vector<pair<int, int>>& colBands) {
    ll F = 0;
    for (auto& rb : rowBands) {
        int h = rb.second - rb.first + 1;
        for (auto& cb : colBands) {
            int w = cb.second - cb.first + 1;
            if (rectDefects(rb.first, rb.second, cb.first, cb.second) > 0) continue;
            ll bestVal = 0;
            for (int i = 1; i <= K; i++)
                if (fits(i, w, h) && V[i] > bestVal) bestVal = V[i];
            F += bestVal;
        }
    }
    return F;
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

    // A row/column is "dirty" only once its defect DENSITY is high (a real
    // scarred band), not merely touched by scattered background noise -- a
    // single stray defect elsewhere in a mostly-clean row should not force
    // that whole row down to singleton bands.
    ll rowThresh = max((ll)1, (ll)(0.25 * N));
    vector<char> rowDirty(N + 1, 0), colDirty(N + 1, 0);
    for (int r = 1; r <= N; r++) rowDirty[r] = (rectDefects(r, r, 1, N) > rowThresh);
    for (int c = 1; c <= N; c++) colDirty[c] = (rectDefects(1, N, c, c) > rowThresh);

    // Candidate clean-region pitches: every square die side in the catalog.
    vector<int> caps;
    for (int i = 1; i <= K; i++) if (A[i] == Bd[i]) caps.push_back(A[i]);
    if (caps.empty()) caps.push_back(1);

    int bestCap = caps[0];
    ll bestF = -1;
    for (int cap : caps) {
        vector<int> hc = buildCuts(N, rowDirty, cap);
        vector<int> vc = buildCuts(N, colDirty, cap);
        ll F = evalAssignment(bandsFromCuts(hc, N), bandsFromCuts(vc, N));
        if (F > bestF) { bestF = F; bestCap = cap; }
    }

    vector<int> hcut = buildCuts(N, rowDirty, bestCap);
    vector<int> vcut = buildCuts(N, colDirty, bestCap);
    vector<pair<int, int>> rowBands = bandsFromCuts(hcut, N);
    vector<pair<int, int>> colBands = bandsFromCuts(vcut, N);

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
