// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// THE INSIGHT: the single 1x1 mat is a parity defect. Placing it anywhere but
// the court's own centre of symmetry forecloses a globally mirror-symmetric
// seam graph. So: reserve the centre cell for the defect FIRST, then tile the
// rest of the court as mirror-symmetric block pairs -- build one block's
// running-bond pattern, and get its mirror partner for free by reflecting
// every domino through the centre (guaranteed valid because pillars are
// always placed as point-symmetric pairs too). Alternating each block's
// internal orientation (row-offset horizontal vs column-offset vertical) by
// (block-row + block-col) parity is itself point-symmetric (odd block counts
// preserve that parity under reflection), so it adds real local orientation
// mixing without breaking the global symmetry. A thin uncovered "grout line"
// separates blocks, which is what makes mixing orientations safe (a court
// with mixed regions but zero gap between them generically breaks the
// four-corner rule at the seam) -- coverage is traded for entropy + symmetry.

static int H, W, cr, cc;
static vector<string> g;
static vector<vector<int>> matId;
static vector<tuple<char,int,int>> out;
static int mid = 0;

static bool tryPlace(char type, int r, int c) {
    int r2 = r, c2 = c;
    if (type == 'H') c2 = c + 1; else r2 = r + 1;
    if (r < 0 || r2 >= H || c < 0 || c2 >= W) return false;
    if ((r == cr && c == cc) || (r2 == cr && c2 == cc)) return false;
    if (g[r][c] != '.' || g[r2][c2] != '.') return false;
    if (matId[r][c] != -1 || matId[r2][c2] != -1) return false;
    matId[r][c] = mid; matId[r2][c2] = mid; mid++;
    out.push_back({type, r, c});
    return true;
}

static void mirrorPlace(char type, int r, int c) {
    int r2 = r, c2 = c;
    if (type == 'H') c2 = c + 1; else r2 = r + 1;
    int mr = H - 1 - r, mc = W - 1 - c, mr2 = H - 1 - r2, mc2 = W - 1 - c2;
    char mtype; int ar, ac;
    if (mr == mr2) { mtype = 'H'; ar = mr; ac = min(mc, mc2); }
    else { mtype = 'V'; ar = min(mr, mr2); ac = mc; }
    tryPlace(mtype, ar, ac);
}

// fill block [r0,r1] x [c0,c1] with a fixed-phase running-bond pattern; also
// mirror-copy every domino placed unless this block is its own mirror image.
static void fillBlock(int r0, int r1, int c0, int c1, bool vertMode, bool selfMirror) {
    if (vertMode) {
        for (int c = c0; c <= c1; c++) {
            int start = r0 + ((c - c0) % 2);
            for (int r = start; r + 1 <= r1; r += 2) {
                if (tryPlace('V', r, c) && !selfMirror) mirrorPlace('V', r, c);
            }
        }
    } else {
        for (int r = r0; r <= r1; r++) {
            int start = c0 + ((r - r0) % 2);
            for (int c = start; c + 1 <= c1; c += 2) {
                if (tryPlace('H', r, c) && !selfMirror) mirrorPlace('H', r, c);
            }
        }
    }
}

static vector<pair<int,int>> buildSegments(int n, int center, int half) {
    half = max(1, min(half, min(center, n - 1 - center)));
    vector<pair<int,int>> upper; // ascending, strictly above the centre segment
    int pos = center + half + 1 + 1; // +1 gap
    while (pos <= n - 1) {
        int hi = min(pos + 2 * half, n - 1);
        upper.push_back({pos, hi});
        pos = hi + 1 + 1;
    }
    vector<pair<int,int>> segs;
    for (int i = (int)upper.size() - 1; i >= 0; i--) {
        auto [lo, hi] = upper[i];
        segs.push_back({n - 1 - hi, n - 1 - lo});
    }
    segs.push_back({center - half, center + half});
    for (auto &s : upper) segs.push_back(s);
    return segs;
}

int main() {
    scanf("%d %d", &H, &W);
    g.resize(H);
    for (int r = 0; r < H; r++) { char buf[512]; scanf("%s", buf); g[r] = buf; }
    cr = (H - 1) / 2; cc = (W - 1) / 2;
    matId.assign(H, vector<int>(W, -1));

    auto rowSegs = buildSegments(H, cr, 3);
    auto colSegs = buildSegments(W, cc, 3);
    int nR = (int)rowSegs.size(), nC = (int)colSegs.size();

    vector<vector<bool>> done(nR, vector<bool>(nC, false));
    for (int bi = 0; bi < nR; bi++) {
        for (int bj = 0; bj < nC; bj++) {
            if (done[bi][bj]) continue;
            int mi = nR - 1 - bi, mj = nC - 1 - bj;
            bool self = (mi == bi && mj == bj);
            bool vertMode = ((bi + bj) % 2 == 1);
            fillBlock(rowSegs[bi].first, rowSegs[bi].second,
                      colSegs[bj].first, colSegs[bj].second, vertMode, self);
            done[bi][bj] = true;
            done[mi][mj] = true;
        }
    }

    out.push_back({'S', cr, cc});

    printf("%d\n", (int)out.size());
    for (auto &t : out) {
        auto [ty, r, c] = t;
        printf("%c %d %d\n", ty, r, c);
    }
    return 0;
}
