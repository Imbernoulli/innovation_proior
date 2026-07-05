#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int W, H, K, T;
vector<vector<char>> blockedGrid;                 // [r][c] true = crevasse
vector<vector<pair<int,int>>> shapes;             // base footprints
set<vector<pair<int,int>>> allowed;               // all normalized orientations of all shapes

// normalize a set of cells: translate so min row/col are 0, then sort.
static vector<pair<int,int>> normalize(vector<pair<int,int>> cells) {
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : cells) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : cells) { p.first -= mr; p.second -= mc; }
    sort(cells.begin(), cells.end());
    return cells;
}

// generate the 8 dihedral orientations of a footprint (normalized).
static void addOrientations(const vector<pair<int,int>>& base) {
    for (int refl = 0; refl < 2; refl++) {
        for (int rot = 0; rot < 4; rot++) {
            vector<pair<int,int>> cur;
            for (auto& p : base) {
                int r = p.first, c = p.second;
                if (refl) c = -c;
                for (int t = 0; t < rot; t++) { int nr = c, nc = -r; r = nr; c = nc; }
                cur.push_back({r, c});
            }
            allowed.insert(normalize(cur));
        }
    }
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    W = inf.readInt();
    H = inf.readInt();
    K = inf.readInt();
    blockedGrid.assign(H, vector<char>(W, 0));
    for (int i = 0; i < K; i++) {
        int r = inf.readInt();
        int c = inf.readInt();
        blockedGrid[r][c] = 1;
    }
    T = inf.readInt();
    shapes.resize(T);
    for (int i = 0; i < T; i++) {
        int sz = inf.readInt();
        shapes[i].resize(sz);
        for (int j = 0; j < sz; j++) {
            int r = inf.readInt();
            int c = inf.readInt();
            shapes[i][j] = {r, c};
        }
        addOrientations(shapes[i]);
    }

    // ---- internal baseline B: naive coarse-lattice deployment of footprint 0
    // (given orientation) at anchors spaced by (bboxH+2, bboxW+2), placed wherever
    // it lands on solid cells. This is a deliberately sparse reference covering. ----
    ll B;
    {
        vector<pair<int,int>> s0 = normalize(shapes[0]);
        int bh = 0, bw = 0;
        for (auto& p : s0) { bh = max(bh, p.first); bw = max(bw, p.second); }
        int SR = bh + 3, SC = bw + 3;   // spacing >= bbox + 2 (copies never overlap)
        ll cover = 0;
        for (int r = 0; r < H; r += SR) for (int c = 0; c < W; c += SC) {
            bool fits = true;
            for (auto& p : s0) {
                int rr = r + p.first, cc = c + p.second;
                if (rr < 0 || rr >= H || cc < 0 || cc >= W) { fits = false; break; }
                if (blockedGrid[rr][cc]) { fits = false; break; }
            }
            if (fits) cover += (ll)s0.size();
        }
        // fallback guarantees B>0 even on pathological tiny inputs.
        B = (cover > 0) ? cover : (ll)shapes[0].size();
    }

    // ---- read & validate participant's placements ----
    vector<vector<char>> occ(H, vector<char>(W, 0));
    ll F = 0;
    int P = ouf.readInt(0, W * H, "P");
    for (int i = 0; i < P; i++) {
        int sz = ouf.readInt(1, 5, "sz");
        vector<pair<int,int>> cells(sz);
        for (int j = 0; j < sz; j++) {
            int r = ouf.readInt(0, H - 1, "r");
            int c = ouf.readInt(0, W - 1, "c");
            cells[j] = {r, c};
        }
        // solid + disjoint (also catches within-placement duplicates via occ marking)
        for (auto& p : cells) {
            int r = p.first, c = p.second;
            if (blockedGrid[r][c]) quitf(_wa, "placement %d covers crevasse cell (%d,%d)", i, r, c);
            if (occ[r][c]) quitf(_wa, "placement %d overlaps an occupied cell (%d,%d)", i, r, c);
        }
        // congruent to some catalogue footprint under rotation/reflection?
        vector<pair<int,int>> norm = normalize(cells);
        if (!allowed.count(norm))
            quitf(_wa, "placement %d is not a legal rotation/reflection of any footprint", i);
        // commit
        for (auto& p : cells) occ[p.first][p.second] = 1;
        F += sz;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);
    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
