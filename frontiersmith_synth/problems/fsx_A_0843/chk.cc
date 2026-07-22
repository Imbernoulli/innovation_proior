#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;

// -----------------------------------------------------------------------------
// Checker / scorer for "Steamworks Deck: Cold-Rail Block Stowage".
//
// Input:  W H N ITERS ; then per block i=1..N:  w_i k_i  \n  dx_1 dy_1 ... dx_k dy_k
// Output: N lines  x_i y_i r_i  (anchor + rotation in {0,1,2,3}) for each block,
//         in input order.
//
// Feasibility: every block's rotated+translated cells lie in [0,W)x[0,H); no two
// blocks' cells overlap; r in {0,1,2,3}; no trailing tokens.
//
// Objective (MIN): build source array S (floor(1000*w_i/k_i) added per cell of
// block i), run ITERS steps of integer Jacobi diffusion with the rim (x=0,x=W-1,
// y=0,y=H-1) clamped to temperature 0, F = max temperature reached anywhere.
//
// Baseline B (checker-computed): shelf-pack the N blocks, in INPUT order, at
// rotation 0 (fill a row left-to-right until the next block's bbox would cross
// x=W, then start a new row above the tallest block used so far) -- this is
// exactly what solutions/trivial.cpp reproduces -> ratio 0.1.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int W, H, N, ITERS;
vector<ll> watt;
vector<vector<pii>> shape; // shape[i] = normalized (min dx=0,min dy=0) cell offsets

// rotate+re-normalize a shape's offsets for rotation r in {0,1,2,3}
static vector<pii> rotateNorm(const vector<pii>& cells, int r){
    vector<pii> out;
    out.reserve(cells.size());
    for (auto& c : cells){
        int dx = c.first, dy = c.second, nx, ny;
        switch (r){
            case 0: nx = dx;  ny = dy;  break;
            case 1: nx = dy;  ny = -dx; break;
            case 2: nx = -dx; ny = -dy; break;
            default: nx = -dy; ny = dx; break; // r == 3
        }
        out.push_back({nx, ny});
    }
    int mnx = INT_MAX, mny = INT_MAX;
    for (auto& c : out){ mnx = min(mnx, c.first); mny = min(mny, c.second); }
    for (auto& c : out){ c.first -= mnx; c.second -= mny; }
    return out;
}

static void bbox(const vector<pii>& cells, int& bw, int& bh){
    int mx = 0, my = 0;
    for (auto& c : cells){ mx = max(mx, c.first); my = max(my, c.second); }
    bw = mx + 1; bh = my + 1;
}

// Baseline construction: shelf-pack the N blocks (input order, rotation 0)
// into rows exactly as a naive packer would, but then CENTER the whole
// packed block both vertically (rows) and horizontally (within each row) in
// the deck, instead of anchoring it at the corner. A corner anchor would
// accidentally hug the rim on its own (a compact stack of small blocks
// started at (0,0) mostly lies on/near the rim whenever there is a lot of
// spare deck around it) which is not a fair "physics-ignorant" reference.
// Centering removes that accident: this construction ends up squarely in the
// worst (most interior) part of the deck, exactly what a solver that never
// looks at the cold rim would produce on average.
static vector<pii> centeredShelfPack(int W, int H, int N, const vector<vector<pii>>& shp){
    vector<int> bw(N + 1), bh(N + 1);
    for (int i = 1; i <= N; i++) bbox(shp[i], bw[i], bh[i]);
    vector<int> rowOf(N + 1), localX(N + 1);
    vector<int> rowWidth, rowHeight;
    int cursor_x = 0, row = 0, row_h = 0;
    for (int i = 1; i <= N; i++){
        if (cursor_x + bw[i] > W){
            rowWidth.push_back(cursor_x);
            rowHeight.push_back(row_h);
            row++; cursor_x = 0; row_h = 0;
        }
        rowOf[i] = row; localX[i] = cursor_x;
        cursor_x += bw[i];
        row_h = max(row_h, bh[i]);
    }
    rowWidth.push_back(cursor_x);
    rowHeight.push_back(row_h);
    int numRows = row + 1;
    int totalH = 0; for (int h : rowHeight) totalH += h;
    int topY = (H - totalH) / 2; if (topY < 0) topY = 0;
    vector<int> rowY(numRows);
    rowY[0] = topY;
    for (int r = 1; r < numRows; r++) rowY[r] = rowY[r - 1] + rowHeight[r - 1];
    vector<pii> anchors(N + 1);
    for (int i = 1; i <= N; i++){
        int r = rowOf[i];
        int xoff = (W - rowWidth[r]) / 2; if (xoff < 0) xoff = 0;
        anchors[i] = {xoff + localX[i], rowY[r]};
    }
    return anchors;
}

// run ITERS steps of integer Jacobi diffusion on a W*H grid with rim clamped
// to 0; S is the WxH source array (row-major, idx = y*W+x). Returns peak T.
static ll diffuse(const vector<ll>& S){
    vector<ll> cur(W * H, 0), nxt(W * H, 0);
    for (int it = 0; it < ITERS; it++){
        nxt = cur; // boundary stays whatever cur had there (always 0)
        for (int y = 1; y <= H - 2; y++){
            for (int x = 1; x <= W - 2; x++){
                int idx = y * W + x;
                ll s = cur[idx - 1] + cur[idx + 1] + cur[idx - W] + cur[idx + W] + 4 * S[idx];
                nxt[idx] = s / 4;
            }
        }
        cur.swap(nxt);
    }
    ll peak = 0;
    for (ll v : cur) peak = max(peak, v);
    return peak;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    W = inf.readInt();
    H = inf.readInt();
    N = inf.readInt();
    ITERS = inf.readInt();
    watt.resize(N + 1);
    shape.resize(N + 1);
    for (int i = 1; i <= N; i++){
        watt[i] = inf.readLong();
        int k = inf.readInt();
        vector<pii> cells(k);
        for (int j = 0; j < k; j++){
            int dx = inf.readInt();
            int dy = inf.readInt();
            cells[j] = {dx, dy};
        }
        shape[i] = cells;
    }

    // ---- internal baseline B: centered shelf-pack in input order, rotation 0 ----
    vector<ll> Sbase(W * H, 0);
    {
        vector<pii> anchors = centeredShelfPack(W, H, N, shape);
        for (int i = 1; i <= N; i++){
            int k = (int)shape[i].size();
            ll add = watt[i] * 1000LL / k;
            for (auto& c : shape[i]){
                int ax = anchors[i].first + c.first, ay = anchors[i].second + c.second;
                if (ax >= 0 && ax < W && ay >= 0 && ay < H) Sbase[ay * W + ax] += add;
            }
        }
    }
    ll B = diffuse(Sbase);
    if (B <= 0) B = 1; // generator guarantees a positive baseline peak

    // ---- read + validate the participant's placement ----
    vector<vector<int>> occ(H, vector<int>(W, -1));
    vector<ll> Ssol(W * H, 0);
    for (int i = 1; i <= N; i++){
        int x = ouf.readInt(-1000000, 1000000, "x");
        int y = ouf.readInt(-1000000, 1000000, "y");
        int r = ouf.readInt(0, 3, "rotation");
        vector<pii> cells = rotateNorm(shape[i], r);
        int k = (int)cells.size();
        ll add = watt[i] * 1000LL / k;
        for (auto& c : cells){
            int ax = x + c.first, ay = y + c.second;
            if (ax < 0 || ax >= W || ay < 0 || ay >= H)
                quitf(_wa, "block %d cell (%d,%d) out of deck bounds", i, ax, ay);
            if (occ[ay][ax] != -1)
                quitf(_wa, "block %d overlaps block %d at (%d,%d)", i, occ[ay][ax], ax, ay);
            occ[ay][ax] = i;
            Ssol[ay * W + ax] += add;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after block %d", N);

    ll F = diffuse(Ssol);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
