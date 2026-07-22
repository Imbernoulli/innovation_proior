#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Dicing a Scarred Wafer into Unequal Chips".  family:
// defect-aware-line-dicing.
//
// Catalog design (every test): die 1 = base unit (1,1); a handful of SQUARE
// dies of increasing side up to Smax = max(2, N/6); one ELONGATED die
// (1, Smax) that only fits into height-1 or width-1 bands. Values follow
// value(a,b) = round((a*b)^1.4 * 5 * jitter) -- strictly superadditive in
// area (a big die is worth far more than the sum of small dies covering the
// same area), so the biggest square die is always the single highest-value
// catalog entry and is what a naive "pick the best die, tile uniformly"
// greedy commits to everywhere on the wafer.
//
// PLANTED TRAPS (testId 4,5,6,9 -- 4/10 cases): defect cells are packed
// almost solidly into a handful of whole ROWS and/or COLUMNS, positioned at
// an OFFSET from multiples of Smax (e.g. k*Smax - Smax/2) so that greedy's
// fixed uniform Smax x Smax pitch (anchored at the corner) straddles every
// dirty row/column -- almost every uniform cell it creates contains a
// defective cell and is wasted outright. A solver that instead reads the
// per-row / per-column defect pattern and re-derives the cut positions
// (isolating dirty rows/columns into their own thin bands, keeping the big
// pitch only where a run is provably all-clean) salvages nearly all of the
// clean area. testId 6 and 9 additionally leave large untouched clean
// blocks between the dense bands (a "needle": a big contiguous clean region
// only a line-placement search that looks at the WHOLE row/column profile,
// not a fixed pitch, will recover intact).
// -----------------------------------------------------------------------------

int N;
vector<vector<char>> defect;

static void markCell(int r, int c) {
    if (r >= 1 && r <= N && c >= 1 && c <= N) defect[r][c] = 1;
}

static void sparse(double p) {
    for (int r = 1; r <= N; r++)
        for (int c = 1; c <= N; c++)
            if (rnd.next(0.0, 1.0) < p) defect[r][c] = 1;
}

static void blobs(int nb, int rmin, int rmax, double fillp) {
    for (int b = 0; b < nb; b++) {
        int cr = rnd.next(1, N), cc = rnd.next(1, N);
        int rad = rnd.next(rmin, rmax);
        for (int r = max(1, cr - rad); r <= min(N, cr + rad); r++)
            for (int c = max(1, cc - rad); c <= min(N, cc + rad); c++) {
                int dr = r - cr, dc = c - cc;
                if (dr * dr + dc * dc <= rad * rad && rnd.next(0.0, 1.0) < fillp)
                    defect[r][c] = 1;
            }
    }
}

static void denseRows(const vector<int>& rows, double fillp) {
    for (int r : rows) {
        if (r < 1 || r > N) continue;
        for (int c = 1; c <= N; c++)
            if (rnd.next(0.0, 1.0) < fillp) markCell(r, c);
    }
}

static void denseCols(const vector<int>& cols, double fillp) {
    for (int c : cols) {
        if (c < 1 || c > N) continue;
        for (int r = 1; r <= N; r++)
            if (rnd.next(0.0, 1.0) < fillp) markCell(r, c);
    }
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    switch (t) {
        case 1: N = 8; break;
        case 2: N = 16; break;
        case 3: N = 24; break;
        case 4: N = 40; break;
        case 5: N = 40; break;
        case 6: N = 55; break;
        case 7: N = 70; break;
        case 8: N = 95; break;
        case 9: N = 120; break;
        default: N = 150; break;
    }
    defect.assign(N + 2, vector<char>(N + 2, 0));

    // ---- die catalog ----
    int Smax = max(2, N / 6);
    int mid = Smax / 2;
    vector<int> squares;
    squares.push_back(2);
    if (mid >= 2 && mid != 2 && mid != Smax) squares.push_back(mid);
    if (Smax != 2) squares.push_back(Smax);
    sort(squares.begin(), squares.end());
    squares.erase(unique(squares.begin(), squares.end()), squares.end());

    vector<pair<int, int>> dims;
    dims.push_back({1, 1});
    for (int s : squares) dims.push_back({s, s});
    dims.push_back({1, Smax});
    int K = (int)dims.size();

    vector<ll> val(K);
    for (int i = 0; i < K; i++) {
        double area = (double)dims[i].first * (double)dims[i].second;
        double raw = pow(area, 1.4) * 5.0;
        double jitter = 1.0 + rnd.next(-50, 50) / 1000.0; // +-5%
        ll v = (ll)llround(raw * jitter);
        if (v < 1) v = 1;
        val[i] = v;
    }

    // ---- defect pattern per test type ----
    switch (t) {
        case 1:
            sparse(0.05);
            break;
        case 2:
            sparse(0.05);
            break;
        case 3:
            blobs(3, 2, 5, 0.75);
            break;
        case 4: { // TRAP: dense row bands offset from Smax multiples
            vector<int> rows;
            for (int k = 1; (ll)k * Smax < N; k++) {
                int r = k * Smax - Smax / 2;
                if (r < 1) r = 1;
                rows.push_back(r);
            }
            denseRows(rows, 0.9);
            break;
        }
        case 5: { // TRAP: dense column bands offset from Smax multiples
            vector<int> cols;
            for (int k = 1; (ll)k * Smax < N; k++) {
                int c = k * Smax - Smax / 2;
                if (c < 1) c = 1;
                cols.push_back(c);
            }
            denseCols(cols, 0.9);
            break;
        }
        case 6: { // TRAP + needle: cross-hatch bands, big clean gaps between them
            vector<int> rows = {(int)(N * 0.30), (int)(N * 0.68)};
            vector<int> cols = {(int)(N * 0.35), (int)(N * 0.72)};
            denseRows(rows, 0.85);
            denseCols(cols, 0.85);
            break;
        }
        case 7:
            blobs(5, 3, 7, 0.7);
            break;
        case 8:
            sparse(0.15);
            break;
        case 9: { // TRAP + needle: structured offset bands on BOTH axes, thinned
            vector<int> rowsAll, colsAll;
            for (int k = 0; (ll)k * Smax + Smax / 2 < N; k++)
                rowsAll.push_back(k * Smax + Smax / 2 + 1);
            for (int k = 0; (ll)k * Smax + Smax / 3 < N; k++)
                colsAll.push_back(k * Smax + Smax / 3 + 1);
            vector<int> rows, cols;
            for (size_t i = 0; i < rowsAll.size(); i++) if (i % 2 == 0) rows.push_back(rowsAll[i]);
            for (size_t i = 0; i < colsAll.size(); i++) if (i % 2 == 1) cols.push_back(colsAll[i]);
            denseRows(rows, 0.9);
            denseCols(cols, 0.9);
            break;
        }
        default: { // testId 10: largest, adversarial mix (scatter + bands)
            sparse(0.04);
            vector<int> rows = {(int)(N * 0.20), (int)(N * 0.50), (int)(N * 0.81)};
            vector<int> cols = {(int)(N * 0.28), (int)(N * 0.63)};
            denseRows(rows, 0.85);
            denseCols(cols, 0.85);
            break;
        }
    }

    // Safety: never let the whole wafer go defective (baseline B must stay positive
    // with real headroom); clear a small guaranteed-clean corner patch if needed.
    int cnt = 0;
    for (int r = 1; r <= N; r++) for (int c = 1; c <= N; c++) if (defect[r][c]) cnt++;
    if (cnt > N * N - 9) {
        for (int r = 1; r <= min(3, N); r++)
            for (int c = 1; c <= min(3, N); c++)
                defect[r][c] = 0;
    }

    vector<pair<int, int>> defs;
    for (int r = 1; r <= N; r++)
        for (int c = 1; c <= N; c++)
            if (defect[r][c]) defs.push_back({r, c});

    printf("%d %d\n", N, K);
    for (int i = 0; i < K; i++) printf("%d %d %lld\n", dims[i].first, dims[i].second, val[i]);
    printf("%d\n", (int)defs.size());
    for (auto& pr : defs) printf("%d %d\n", pr.first, pr.second);

    return 0;
}
