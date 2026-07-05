#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int S = inf.readInt();
    int G = inf.readInt();
    int T = inf.readInt();
    int Cfull = inf.readInt();
    int Cmin  = inf.readInt();
    int Drain = inf.readInt();
    int Recharge = inf.readInt();

    vector<ll> D(G + 1);
    for (int g = 1; g <= G; g++) D[g] = inf.readLong();

    vector<vector<int>> cov(S + 1);       // sensor -> targets it reaches
    vector<ll> c(S + 1);
    ll sumCost = 0;
    for (int s = 1; s <= S; s++) {
        c[s] = inf.readInt();
        sumCost += c[s];
        int k = inf.readInt();
        cov[s].resize(k);
        for (int j = 0; j < k; j++) cov[s][j] = inf.readInt(1, G, "target");
    }

    // internal baseline: all-on every round (always feasible by construction).
    ll B = (ll)T * sumCost;
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---------- read participant schedule & simulate charges exactly ----------
    vector<ll> h(S + 1, Cfull);           // current charge
    ll F = 0;
    vector<ll> cover(G + 1, 0);           // reused per round
    vector<char> on(S + 1, 0);
    vector<int> onList;

    for (int r = 1; r <= T; r++) {
        int m = ouf.readInt(0, S, "m_r");
        onList.clear();
        for (int i = 0; i < m; i++) {
            int idx = ouf.readInt(1, S, "sensorIndex");
            if (on[idx]) quitf(_wa, "round %d: sensor %d switched on twice", r, idx);
            on[idx] = 1;
            onList.push_back(idx);
            F += c[idx];
        }
        // deposit pulses (= current charge) into reached targets
        for (int s : onList)
            for (int t : cov[s]) cover[t] += h[s];
        // feasibility: every target's coverage this round must meet demand
        for (int g = 1; g <= G; g++)
            if (cover[g] < D[g])
                quitf(_wa, "round %d: target %d coverage %lld < demand %lld",
                      r, g, cover[g], D[g]);
        // charge dynamics: drained on-sensors decay (floor Cmin); rest recharge (cap Cfull)
        for (int s = 1; s <= S; s++) {
            if (on[s]) h[s] = max((ll)Cmin, h[s] - Drain);
            else       h[s] = min((ll)Cfull, h[s] + Recharge);
        }
        // reset per-round scratch
        for (int s : onList) on[s] = 0;
        for (int g = 1; g <= G; g++) cover[g] = 0;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d rounds", T);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
