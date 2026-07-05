#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int N, M, W, H, R;
static vector<ll> w, h, p;
static vector<vector<int>> nets;

// F = WIRE + THERM for a placement (x[i],y[i]) lower-left corners.
static ll objective(const vector<ll>& x, const vector<ll>& y) {
    // centers in doubled coords
    vector<ll> cx(N), cy(N);
    for (int i = 0; i < N; i++) { cx[i] = 2 * x[i] + w[i]; cy[i] = 2 * y[i] + h[i]; }
    ll wire = 0;
    for (auto& s : nets) {
        ll xmn = LLONG_MAX, xmx = LLONG_MIN, ymn = LLONG_MAX, ymx = LLONG_MIN;
        for (int b : s) {
            xmn = min(xmn, cx[b]); xmx = max(xmx, cx[b]);
            ymn = min(ymn, cy[b]); ymx = max(ymx, cy[b]);
        }
        wire += (xmx - xmn) + (ymx - ymn);
    }
    ll therm = 0;
    for (int i = 0; i < N; i++) {
        if (p[i] == 0) continue;
        for (int j = i + 1; j < N; j++) {
            if (p[j] == 0) continue;
            ll d = llabs(cx[i] - cx[j]) + llabs(cy[i] - cy[j]);
            if (d < R) therm += p[i] * p[j] * (ll)(R - d);
        }
    }
    return wire + therm;
}

// index-order shelf baseline (matches statement + trivial solution)
static ll baseline() {
    vector<ll> x(N), y(N);
    ll cx = 0, cy = 0; ll rowh = 0;
    for (int i = 0; i < N; i++) {
        if (cx + w[i] > W) { cy += rowh; cx = 0; rowh = 0; }
        x[i] = cx; y[i] = cy;
        cx += w[i];
        rowh = max(rowh, h[i]);
    }
    return objective(x, y);
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);
    W = inf.readInt(); H = inf.readInt(); N = inf.readInt(); M = inf.readInt(); R = inf.readInt();
    w.resize(N); h.resize(N); p.resize(N);
    for (int i = 0; i < N; i++) { w[i] = inf.readInt(); h[i] = inf.readInt(); p[i] = inf.readInt(); }
    nets.resize(M);
    for (int e = 0; e < M; e++) {
        int k = inf.readInt();
        nets[e].resize(k);
        for (int j = 0; j < k; j++) nets[e][j] = inf.readInt() - 1;
    }

    // ---- read participant placement (bounded => rejects nan/inf/garbage/out-of-range) ----
    vector<ll> x(N), y(N);
    for (int i = 0; i < N; i++) {
        x[i] = ouf.readLong(0, (ll)W - w[i], format("x[%d]", i + 1).c_str());
        y[i] = ouf.readLong(0, (ll)H - h[i], format("y[%d]", i + 1).c_str());
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // ---- non-overlap (O(N^2), N<=300) ----
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            bool ox = (x[i] < x[j] + w[j]) && (x[j] < x[i] + w[i]);
            bool oy = (y[i] < y[j] + h[j]) && (y[j] < y[i] + h[i]);
            if (ox && oy) quitf(_wa, "blocks %d and %d overlap", i + 1, j + 1);
        }

    ll F = objective(x, y);
    ll B = baseline();
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
