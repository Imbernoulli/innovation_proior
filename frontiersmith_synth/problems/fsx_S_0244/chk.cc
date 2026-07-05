#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Checker/scorer for Single-Rack LIFO Picking Robot.
// Validates: index ranges, at-most-once pickup/delivery, served-iff-both,
// LIFO stack discipline (delivery must be the top tote), rack height <= H.
// Objective F = tour Manhattan length + sum of skipped penalties (minimize).
// Baseline B = sum of all penalties (serve nobody). Ratio = min(1, (B/F)/10).

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int P = inf.readInt(1, 80, "P");
    int H = inf.readInt(1, 12, "H");
    long long x0 = inf.readInt(0, 1000, "x0");
    long long y0 = inf.readInt(0, 1000, "y0");

    vector<long long> px(P+1), py(P+1), dx(P+1), dy(P+1), w(P+1);
    long long B = 0;
    for (int i = 1; i <= P; i++) {
        px[i] = inf.readInt(0, 1000, "px");
        py[i] = inf.readInt(0, 1000, "py");
        dx[i] = inf.readInt(0, 1000, "dx");
        dy[i] = inf.readInt(0, 1000, "dy");
        w[i]  = inf.readInt(1, 20000, "w");
        B += w[i];
    }
    if (B <= 0) B = 1; // safety; w_i>=1 guarantees B>0 anyway

    long long L = ouf.readInt(0, 2LL*P, "L");

    vector<char> pickedUp(P+1, 0), delivered(P+1, 0);
    vector<int> stk;                 // LIFO stack of tote ids
    long long D = 0;
    long long cx = x0, cy = y0;      // current robot position

    auto manh = [](long long ax, long long ay, long long bx, long long by)->long long {
        return llabs(ax-bx) + llabs(ay-by);
    };

    for (long long e = 0; e < L; e++) {
        int t = ouf.readInt(0, 1, "t");
        int i = ouf.readInt(1, P, "i");
        if (t == 0) {
            if (pickedUp[i]) quitf(_wa, "request %d picked up more than once", i);
            pickedUp[i] = 1;
            if ((int)stk.size() >= H)
                quitf(_wa, "rack height exceeded (H=%d) at event %lld", H, e+1);
            stk.push_back(i);
            long long nx = px[i], ny = py[i];
            D += manh(cx, cy, nx, ny);
            cx = nx; cy = ny;
        } else {
            if (delivered[i]) quitf(_wa, "request %d delivered more than once", i);
            if (!pickedUp[i]) quitf(_wa, "request %d delivered before pickup", i);
            if (stk.empty() || stk.back() != i)
                quitf(_wa, "LIFO violation: request %d is not on top of the rack", i);
            delivered[i] = 1;
            stk.pop_back();
            long long nx = dx[i], ny = dy[i];
            D += manh(cx, cy, nx, ny);
            cx = nx; cy = ny;
        }
    }
    if (!stk.empty())
        quitf(_wa, "%d tote(s) still on rack at end (picked up but not delivered)", (int)stk.size());

    // return to dock (if any events happened)
    if (L > 0) D += manh(cx, cy, x0, y0);

    // served iff both appear; single-appearance already impossible (stack empty =>
    // every picked was delivered, and delivery-before-pickup rejected). Still guard.
    long long penalty = 0;
    for (int i = 1; i <= P; i++) {
        bool served = pickedUp[i] && delivered[i];
        if (pickedUp[i] != delivered[i])
            quitf(_wa, "request %d appears exactly once", i);
        if (!served) penalty += w[i];
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing output after tour");

    long long F = D + penalty;
    if (F < 1) F = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc/1000.0, "OK F=%lld B=%lld D=%lld pen=%lld Ratio: %.6f",
          F, B, D, penalty, sc/1000.0);
    return 0;
}
