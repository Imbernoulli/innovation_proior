#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll D(ll ax, ll ay, ll bx, ll by) {
    double ddx = (double)(ax - bx), ddy = (double)(ay - by);
    return (ll)llround(sqrt(ddx * ddx + ddy * ddy));
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    ll Hx = inf.readLong(), Hy = inf.readLong();
    vector<ll> px(N + 1), py(N + 1), dx(N + 1), dy(N + 1), w(N + 1);
    for (int i = 1; i <= N; i++) {
        px[i] = inf.readLong(); py[i] = inf.readLong();
        dx[i] = inf.readLong(); dy[i] = inf.readLong();
        w[i]  = inf.readLong();
    }

    // reference baseline B: serve every task in index order, no skips
    ll B = 0;
    {
        ll cx = Hx, cy = Hy;
        for (int i = 1; i <= N; i++) {
            B += D(cx, cy, px[i], py[i]); cx = px[i]; cy = py[i];
            B += D(cx, cy, dx[i], dy[i]); cx = dx[i]; cy = dy[i];
        }
        B += D(cx, cy, Hx, Hy);
    }
    if (B < 1) B = 1;

    // read participant route
    int M = ouf.readInt(0, 2 * N, "M");
    vector<char> haveP(N + 1, 0), haveD(N + 1, 0);
    vector<int> posP(N + 1, -1), posD(N + 1, -1);
    ll cx = Hx, cy = Hy;
    ll route = 0;
    for (int k = 0; k < M; k++) {
        string tok = ouf.readToken();
        if (tok != "P" && tok != "D")
            quitf(_wa, "visit %d: type must be P or D, got '%s'", k, tok.c_str());
        int i = ouf.readInt(1, N, "idx");
        ll nx, ny;
        if (tok == "P") {
            if (haveP[i]) quitf(_wa, "flower patch of task %d visited twice", i);
            haveP[i] = 1; posP[i] = k; nx = px[i]; ny = py[i];
        } else {
            if (haveD[i]) quitf(_wa, "comb cell of task %d visited twice", i);
            haveD[i] = 1; posD[i] = k; nx = dx[i]; ny = dy[i];
        }
        route += D(cx, cy, nx, ny); cx = nx; cy = ny;
    }
    route += D(cx, cy, Hx, Hy);
    if (!ouf.seekEof()) quitf(_wa, "trailing output after route");

    // feasibility + skip penalties
    ll skip = 0;
    for (int i = 1; i <= N; i++) {
        if (haveP[i] != haveD[i])
            quitf(_wa, "task %d: exactly one of pickup/delivery present", i);
        if (haveP[i] && haveD[i]) {
            if (posP[i] > posD[i])
                quitf(_wa, "task %d: comb cell visited before flower patch", i);
        } else {
            skip += w[i];
        }
    }

    ll F = route + skip;
    if (F < 1) F = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld route=%lld skip=%lld Ratio: %.6f",
          F, B, route, skip, sc / 1000.0);
    return 0;
}
