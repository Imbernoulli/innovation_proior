#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int T, G;
vector<ll> P, b, rate, K, U;
vector<ll> D, W;

// Replay a schedule (x[t][g] in {0,1}) and return total cost.
// Sets *feasible=false if at any step the committed capacity cannot meet the
// wind-adjusted residual demand.  Economic dispatch is cheapest-rate-first,
// which is optimal for a fixed commitment.
ll replayCost(const vector<vector<char>>& x, bool* feasible) {
    *feasible = true;
    ll total = 0;
    vector<char> prev(G, 0);
    // pre-sort generator indices by fuel rate for dispatch
    vector<int> order(G);
    for (int g = 0; g < G; g++) order[g] = g;
    sort(order.begin(), order.end(), [&](int a, int c){
        if (rate[a] != rate[c]) return rate[a] < rate[c];
        return a < c;
    });
    for (int t = 0; t < T; t++) {
        ll capON = 0;
        for (int g = 0; g < G; g++) {
            if (x[t][g]) {
                if (!prev[g]) total += K[g];   // start-up
                total += b[g];                 // no-load
                capON += P[g];
            }
        }
        ll R = D[t] - W[t]; if (R < 0) R = 0;
        if (capON < R) { *feasible = false; }
        // dispatch cheapest first
        ll rem = R;
        for (int idx = 0; idx < G && rem > 0; idx++) {
            int g = order[idx];
            if (!x[t][g]) continue;
            ll use = min(P[g], rem);
            total += use * rate[g];
            rem -= use;
        }
        for (int g = 0; g < G; g++) prev[g] = x[t][g];
    }
    return total;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    T = inf.readInt();
    G = inf.readInt();
    P.resize(G); b.resize(G); rate.resize(G); K.resize(G); U.resize(G);
    for (int g = 0; g < G; g++) {
        P[g]    = inf.readLong();
        b[g]    = inf.readLong();
        rate[g] = inf.readLong();
        K[g]    = inf.readLong();
        U[g]    = inf.readLong();
    }
    D.resize(T); W.resize(T);
    for (int t = 0; t < T; t++) { D[t] = inf.readLong(); W[t] = inf.readLong(); }

    // ---- internal baseline: run EVERY generator for the WHOLE horizon ----
    vector<vector<char>> allOn(T, vector<char>(G, 1));
    bool bfeas = true;
    ll B = replayCost(allOn, &bfeas);
    if (!bfeas || B <= 0) quitf(_fail, "bad instance: baseline infeasible or B=%lld", B);

    // ---- read participant schedule: T rows of G bits (0/1) ----
    vector<vector<char>> x(T, vector<char>(G, 0));
    for (int t = 0; t < T; t++)
        for (int g = 0; g < G; g++)
            x[t][g] = (char)ouf.readInt(0, 1, "onoff");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- validate minimum-up-time (warm-up) constraint ----
    // Every maximal ON-run must have length >= U[g], unless the run reaches the
    // end of the horizon (it is truncated by the horizon boundary).
    for (int g = 0; g < G; g++) {
        int t = 0;
        while (t < T) {
            if (x[t][g]) {
                int a = t;
                while (t < T && x[t][g]) t++;
                int len = t - a;
                bool endsAtHorizon = (t == T);
                if (!endsAtHorizon && len < U[g])
                    quitf(_wa, "generator %d run [%d,%d) has length %d < min-up %lld",
                          g, a, t, len, U[g]);
            } else t++;
        }
    }

    // ---- replay & score ----
    bool feas = true;
    ll F = replayCost(x, &feas);
    if (!feas)
        quitf(_wa, "committed capacity fails to meet demand at some step");
    if (F <= 0) quitf(_wa, "non-positive cost %lld", F);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
