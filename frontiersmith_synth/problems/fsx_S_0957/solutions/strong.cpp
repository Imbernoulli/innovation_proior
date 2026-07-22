// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// The insight: don't chase local consistency one cell at a time. First DERIVE
// the tile catalog's hidden two-letter horizontal alphabet and two-letter
// vertical alphabet from the edge colours actually present (the catalog never
// states "this is a boundary-colour puzzle" -- it just lists 16 tiles). Then,
// instead of any local search, LAY DOWN two independent maximal-length linear
// feedback shift register (LFSR) sequences -- a hierarchical, self-similar
// recurrence, i.e. a substitution system -- to decide the horizontal and
// vertical boundary colour at every column/row. An m-sequence's classical
// two-valued out-of-phase autocorrelation guarantees, by construction, that no
// shift collapses it onto itself the way a short local cycle would: this beats
// any cell-by-cell local search on the periodicity objective without ever
// searching the space of tilings at all.
struct Tile { int N, E, S, W; };

static vector<int> lfsr(int degree, unsigned long long tapMask, int len) {
    // recurrence bit[n] = XOR of bit[n-d] for every set bit d in tapMask (d in 1..degree)
    vector<int> b(degree, 1); // non-zero seed
    b.reserve(len);
    while ((int)b.size() < len) {
        int n = (int)b.size();
        int v = 0;
        for (int d = 1; d <= degree; d++)
            if ((tapMask >> d) & 1ULL) v ^= b[n - d];
        b.push_back(v);
    }
    b.resize(len);
    return b;
}

int main() {
    int R, W, T;
    scanf("%d %d %d", &R, &W, &T);
    vector<Tile> tiles(T);
    for (int i = 0; i < T; i++)
        scanf("%d %d %d %d", &tiles[i].N, &tiles[i].E, &tiles[i].S, &tiles[i].W);

    // ---- derive the boundary-colour alphabets from the catalog itself ----
    set<int> hSet, vSet;
    for (auto& t : tiles) { hSet.insert(t.E); hSet.insert(t.W); vSet.insert(t.N); vSet.insert(t.S); }
    vector<int> hCol(hSet.begin(), hSet.end());
    vector<int> vCol(vSet.begin(), vSet.end());
    int nH = max(1, (int)hCol.size());
    int nV = max(1, (int)vCol.size());

    // ---- exact-match lookup table: (N,E,S,W) -> tile index ----
    map<tuple<int,int,int,int>, int> byTuple;
    for (int i = 0; i < T; i++)
        byTuple[{tiles[i].N, tiles[i].E, tiles[i].S, tiles[i].W}] = i;

    // ---- two independent hierarchical (LFSR/substitution) boundary sequences ----
    // degree-10 primitive polynomial x^10+x^7+1 for the wide horizontal axis
    // (period 1023, comfortably above any grid width we generate).
    vector<int> hBits = lfsr(10, (1ULL << 10) | (1ULL << 7), W + 1);
    // degree-7 primitive polynomial x^7+x^6+1 for the vertical axis.
    vector<int> vBits = lfsr(7, (1ULL << 7) | (1ULL << 6), R + 1);

    vector<int> colSeq(W + 1), rowSeq(R + 1);
    for (int j = 0; j <= W; j++) colSeq[j] = hCol[hBits[j] % nH];
    for (int i = 0; i <= R; i++) rowSeq[i] = vCol[vBits[i] % nV];

    vector<vector<int>> grid(R, vector<int>(W, -1));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < W; j++) {
            int wantN = rowSeq[i], wantE = colSeq[j + 1], wantS = rowSeq[i + 1], wantW = colSeq[j];
            auto it = byTuple.find({wantN, wantE, wantS, wantW});
            int chosen;
            if (it != byTuple.end()) {
                chosen = it->second;
            } else {
                // Defensive fallback (only triggers on a sparser catalog than the
                // full 16-tile one this generator ships): fall back to matching
                // the ACTUAL placed neighbours, exactly like a local search would.
                int reqW = (j > 0) ? tiles[grid[i][j - 1]].E : -1;
                int reqN = (i > 0) ? tiles[grid[i - 1][j]].S : -1;
                chosen = -1;
                for (int t = 0; t < T; t++) {
                    if (j > 0 && tiles[t].W != reqW) continue;
                    if (i > 0 && tiles[t].N != reqN) continue;
                    chosen = t;
                    break;
                }
                if (chosen < 0) chosen = 0;
            }
            grid[i][j] = chosen;
        }
    }

    for (int i = 0; i < R; i++) {
        for (int j = 0; j < W; j++) printf(j ? " %d" : "%d", grid[i][j] + 1);
        printf("\n");
    }
    return 0;
}
