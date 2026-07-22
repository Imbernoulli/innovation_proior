// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// The insight: re-implement the checker's own relaxation law and SEARCH a
// family of tapered trunk profiles -- start at width w0 centered on the
// wall, narrow by 1 block every `ds` rows (ds huge = uniform width, the
// no-taper limit) -- picking whichever profile the simulation itself scores
// highest. This directly reads and exploits K_COND/K_LOSS from the input
// instead of applying any fixed rule of thumb, so it finds the emergent
// width/taper that pure perimeter-maximizing or pure length-maximizing
// constructions cannot.

static int H, W, M, BW, BC;
static double Thot, Tamb, Kcond, Kloss;

static inline bool isBaseNbr(int r, int c) { return r == 1 && c >= BC && c < BC + BW; }

static double relax(const vector<pair<int,int>> &cells, int iters) {
    if (cells.empty()) return 0.0;
    vector<vector<char>> occ(H + 1, vector<char>(W, 0));
    for (auto &pr : cells) occ[pr.first][pr.second] = 1;
    vector<vector<double>> T(H + 1, vector<double>(W, Tamb));
    int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
    for (int it = 0; it < iters; it++) {
        for (auto &pr : cells) {
            int r = pr.first, c = pr.second;
            double sumT = 0.0; int degCond = 0;
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (nr == 0) { if (isBaseNbr(r, c) && dr[d] == -1) { sumT += Thot; degCond++; } continue; }
                if (nr < 1 || nr > H || nc < 0 || nc >= W) continue;
                if (occ[nr][nc]) { sumT += T[nr][nc]; degCond++; }
            }
            int exposed = 4 - degCond;
            T[r][c] = (Kcond * sumT + Kloss * exposed * Tamb) / (Kcond * degCond + Kloss * exposed);
        }
    }
    double F = 0.0;
    for (auto &pr : cells) {
        int r = pr.first, c = pr.second;
        int degCond = 0;
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr == 0) { if (isBaseNbr(r, c) && dr[d] == -1) degCond++; continue; }
            if (nr < 1 || nr > H || nc < 0 || nc >= W) continue;
            if (occ[nr][nc]) degCond++;
        }
        int exposed = 4 - degCond;
        F += Kloss * exposed * (T[r][c] - Tamb);
    }
    return F;
}

// Build a tapered-trunk shape centered on `cb`: width w0 at row 1, narrowing
// by 1 every `ds` rows, at most M cells, filled center-outward per row so a
// budget cutoff mid-row still leaves the trunk fully connected.
static vector<pair<int,int>> buildTaper(int w0, long long ds, int cb) {
    vector<pair<int,int>> cells;
    int used = 0;
    for (int r = 1; r <= H && used < M; r++) {
        long long shrink = (long long) (r - 1) / ds;
        int width = (int) max(1LL, (long long) w0 - shrink);
        width = min(width, W);
        int left = cb - width / 2;
        if (left < 0) left = 0;
        if (left + width - 1 >= W) left = W - width;
        int right = left + width - 1;
        vector<int> cols;
        for (int c = left; c <= right; c++) cols.push_back(c);
        sort(cols.begin(), cols.end(), [&](int a, int b) {
            return abs(a - cb) < abs(b - cb);
        });
        for (int c : cols) {
            if (used >= M) break;
            cells.push_back({r, c});
            used++;
        }
    }
    return cells;
}

int main() {
    int Thot_i, Tamb_i, Kcond_i, Kloss_i;
    cin >> H >> W >> M >> Thot_i >> Tamb_i >> Kcond_i >> Kloss_i >> BW >> BC;
    Thot = Thot_i; Tamb = Tamb_i; Kcond = Kcond_i; Kloss = Kloss_i;
    int cb = BC + BW / 2;

    static const int WSTARTS[] = {1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22};
    static const long long DSTEPS[] = {1, 2, 3, 4, 6, 9, 14, 20, 1000000000LL};
    const int SITER = 800;

    vector<pair<int,int>> best;
    double bestF = -1.0;
    for (int w0 : WSTARTS) {
        if (w0 > W + 5) continue;
        for (long long ds : DSTEPS) {
            vector<pair<int,int>> cand = buildTaper(w0, ds, cb);
            if (cand.empty()) continue;
            double F = relax(cand, SITER);
            if (F > bestF) { bestF = F; best = cand; }
        }
    }
    if (best.empty() && M > 0) best.push_back({1, cb});

    cout << best.size() << "\n";
    for (auto &pr : best) cout << pr.first << " " << pr.second << "\n";
    return 0;
}
