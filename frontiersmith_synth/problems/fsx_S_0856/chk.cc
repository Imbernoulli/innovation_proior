#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int R, C, K;
bool forbidden[16]; // index = w*8 + x*4 + y*2 + z

static inline int patIdx(int w, int x, int y, int z) { return w * 8 + x * 4 + y * 2 + z; }

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    R = inf.readInt();
    C = inf.readInt();
    K = inf.readInt();
    for (int i = 0; i < K; i++) {
        int w = inf.readInt(0, 1), x = inf.readInt(0, 1), y = inf.readInt(0, 1), z = inf.readInt(0, 1);
        forbidden[patIdx(w, x, y, z)] = true;
    }

    // ---- internal baseline B: best single row pattern repeated on every row ----
    // (ignore any row-to-row coupling; this is the "static" reference the problem's
    //  insight -- a genuinely periodic pattern -- must beat.)
    int N = 1 << C;
    int bestRowWeight = -1;
    for (int s = 0; s < N; s++) {
        bool ok = true;
        for (int c = 0; c + 1 < C && ok; c++) {
            int a = (s >> c) & 1, b = (s >> (c + 1)) & 1;
            if (forbidden[patIdx(a, b, a, b)]) ok = false; // self-transition s->s
        }
        if (ok) {
            int w = __builtin_popcount((unsigned)s);
            if (w > bestRowWeight) bestRowWeight = w;
        }
    }
    if (bestRowWeight <= 0) quitf(_fail, "bad instance: no self-consistent row pattern (bestRowWeight=%d)", bestRowWeight);
    ll B = (ll)R * bestRowWeight;

    // ---- read & validate participant's grid ----
    vector<int> rows(R, 0);
    for (int r = 0; r < R; r++) {
        string tok = ouf.readToken();
        if ((int)tok.size() != C) quitf(_wa, "row %d has length %d, expected %d", r, (int)tok.size(), C);
        int mask = 0;
        for (int c = 0; c < C; c++) {
            char ch = tok[c];
            if (ch != '0' && ch != '1') quitf(_wa, "row %d has non-binary character '%c' at column %d", r, ch, c);
            if (ch == '1') mask |= (1 << c);
        }
        rows[r] = mask;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- feasibility: no forbidden 2x2 window anywhere ----
    ll F = 0;
    for (int r = 0; r < R; r++) F += __builtin_popcount((unsigned)rows[r]);
    for (int r = 0; r + 1 < R; r++) {
        int s = rows[r], t = rows[r + 1];
        for (int c = 0; c + 1 < C; c++) {
            int w = (s >> c) & 1, x = (s >> (c + 1)) & 1;
            int y = (t >> c) & 1, z = (t >> (c + 1)) & 1;
            if (forbidden[patIdx(w, x, y, z)])
                quitf(_wa, "forbidden window (%d,%d,%d,%d) at rows %d-%d, cols %d-%d", w, x, y, z, r, r + 1, c, c + 1);
        }
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld bestRowWeight=%d Ratio: %.6f", F, B, bestRowWeight, sc / 1000.0);
    return 0;
}
