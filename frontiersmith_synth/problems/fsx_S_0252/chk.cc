#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int T = inf.readInt();
    ll Q = inf.readLong();
    ll S = inf.readLong();
    ll P = inf.readLong();

    vector<int> solar(T);
    for (int t = 0; t < T; t++) solar[t] = inf.readInt(0, 1);
    vector<ll> pr(T);
    for (int t = 0; t < T; t++) pr[t] = inf.readLong();

    vector<ll> Cap(N), I(N);
    vector<vector<ll>> L(N, vector<ll>(T));
    ll totalLoad = 0;
    for (int i = 0; i < N; i++) {
        Cap[i] = inf.readLong();
        I[i] = inf.readLong();
        for (int t = 0; t < T; t++) { L[i][t] = inf.readLong(); totalLoad += L[i][t]; }
    }

    // internal baseline: all-grid, refill exactly the load each step (always feasible)
    ll B = S + P * totalLoad;
    if (B <= 0) quitf(_fail, "bad instance B=%lld", B);

    // ---- replay & validate participant schedule ----
    ll cost = 0;
    bool prevOn = false;
    vector<ll> x(N);
    for (int t = 0; t < T; t++) {
        int src = ouf.readInt(1, 2, "src");
        ll sum = 0;
        for (int i = 0; i < N; i++) { x[i] = ouf.readLong(0, Q, "x"); sum += x[i]; }
        if (sum > Q) quitf(_wa, "step %d total production %lld exceeds Q=%lld", t + 1, sum, Q);

        bool on = (sum > 0);
        if (on) {
            if (src == 1 && solar[t] != 1)
                quitf(_wa, "step %d uses spot power but solar unavailable", t + 1);
            ll c = (src == 1 ? pr[t] : P);
            cost += c * sum;
            if (!prevOn) cost += S;
        }
        for (int i = 0; i < N; i++) {
            ll v = I[i] + x[i];
            if (v > Cap[i]) v = Cap[i];
            v -= L[i][t];
            if (v < 0)
                quitf(_wa, "step %d room %d reserve %lld < 0 (vaccine spoilage)", t + 1, i + 1, v);
            I[i] = v;
        }
        prevOn = on;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = cost;
    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
