#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int Z, T, C, s;
vector<ll> p;
vector<ll> Cap, init_;
vector<vector<ll>> h;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    Z = inf.readInt();
    T = inf.readInt();
    C = inf.readInt();
    s = inf.readInt();
    p.assign(T, 0);
    for (int t = 0; t < T; t++) p[t] = inf.readInt();
    Cap.assign(Z, 0);
    for (int j = 0; j < Z; j++) Cap[j] = inf.readInt();
    init_.assign(Z, 0);
    for (int j = 0; j < Z; j++) init_[j] = inf.readInt();
    h.assign(Z, vector<ll>(T, 0));
    for (int j = 0; j < Z; j++)
        for (int t = 0; t < T; t++) h[j][t] = inf.readInt();

    // sanity: init in range
    for (int j = 0; j < Z; j++)
        if (init_[j] < 0 || init_[j] > Cap[j])
            quitf(_fail, "bad instance: init out of range at zone %d", j + 1);

    // ---- internal baseline B: reactive schedule x[j][t] = h[j][t] ----
    ll B = 0;
    for (int t = 0; t < T; t++) {
        ll X = 0;
        for (int j = 0; j < Z; j++) X += h[j][t];
        if (X > C) quitf(_fail, "bad instance: reactive infeasible at step %d (%lld>%d)", t, X, C);
        B += p[t] * X + (X > 0 ? (ll)s : 0);
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read participant schedule ----
    vector<vector<ll>> x(Z, vector<ll>(T, 0));
    for (int j = 0; j < Z; j++)
        for (int t = 0; t < T; t++)
            x[j][t] = ouf.readInt(0, C, "x");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- capacity feasibility ----
    for (int t = 0; t < T; t++) {
        ll X = 0;
        for (int j = 0; j < Z; j++) X += x[j][t];
        if (X > C)
            quitf(_wa, "plant capacity exceeded at step %d: %lld > %d", t, X, C);
    }

    // ---- thermal feasibility (replay dynamics per zone) ----
    for (int j = 0; j < Z; j++) {
        ll theta = init_[j];
        for (int t = 0; t < T; t++) {
            theta = theta + h[j][t] - x[j][t];
            if (theta < 0)
                quitf(_wa, "zone %d overcooled below 0 at step %d (theta=%lld)", j + 1, t, theta);
            if (theta > Cap[j])
                quitf(_wa, "zone %d overheated above %lld at step %d (theta=%lld)", j + 1, Cap[j], t, theta);
        }
    }

    // ---- objective F ----
    ll F = 0;
    for (int t = 0; t < T; t++) {
        ll X = 0;
        for (int j = 0; j < Z; j++) X += x[j][t];
        F += p[t] * X + (X > 0 ? (ll)s : 0);
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
