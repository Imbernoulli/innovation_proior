#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);   // testId 1..10, difficulty ladder

    // ---- size ladder ----
    int Q = 6 + 2 * t;                 // 8 .. 26
    int nBg = 3 + t + (t / 2);         // background (filler) guilds: 4 .. 18
    int n = nBg + 2;                   // + planted pair D (dominant), E (shadow)

    // ---- planted V: two-piece-linear true value, saturation point varies
    //      across the middle third of the quality range (never a fixed
    //      guessable fraction of Q) ----
    int q0V = rnd.next(max(1, Q / 3), max(1, (2 * Q) / 3));
    q0V = max(1, min(Q - 1, q0V));
    ll slope1 = 40 + 3 * t + rnd.next(-5, 5);        // steep pre-kink value slope
    slope1 = max(20LL, slope1);
    // post-kink slope: a small fraction of slope1 (genuine diminishing returns),
    // rotates through a few regimes across the ladder for variety.
    ll slope2;
    int regime = t % 3;
    if (regime == 0) slope2 = 0;
    else if (regime == 1) slope2 = max(1LL, slope1 / 20);
    else slope2 = max(1LL, slope1 / 10);

    vector<ll> V(Q + 1);
    for (int q = 0; q <= Q; q++) {
        if (q <= q0V) V[q] = slope1 * q;
        else V[q] = slope1 * q0V + slope2 * (q - q0V);
    }

    // ---- planted D: cheap uniformly, the "obvious" star performer ----
    ll cheapD = max(1LL, (ll)llround(slope1 * 0.20)) + rnd.next(0, 2);
    // ---- planted E: steep pre-kink (near value-slope, thin margin) but the
    //      CHEAPEST producer of quality past the kink -- an advantage a
    //      scoring rule that only mirrors V (flat past q0V) never rewards.
    ll steepE = max(cheapD + 1, (ll)llround(slope1 * 0.85)) + rnd.next(-3, 3);
    steepE = max(steepE, cheapD + 1);
    ll cheapE = max(1LL, (ll)llround(cheapD * 0.45));
    if (cheapE >= cheapD) cheapE = max(1LL, cheapD - 1);

    vector<ll> cD(Q + 1), cE(Q + 1);
    for (int q = 0; q <= Q; q++) {
        cD[q] = cheapD * (ll)q;
        if (q <= q0V) cE[q] = steepE * (ll)q;
        else cE[q] = steepE * (ll)q0V + cheapE * (ll)(q - q0V);
    }

    // ---- background guilds: always dominated by E's pre-kink slope, so the
    //      D-vs-E dynamic drives the winner/runner-up in most parameter choices ----
    vector<vector<ll>> bg(nBg, vector<ll>(Q + 1));
    for (int j = 0; j < nBg; j++) {
        ll mid = steepE + 1 + rnd.next(0, (int)max(1LL, steepE * 6 / 10));
        for (int q = 0; q <= Q; q++) bg[j][q] = mid * (ll)q;
    }

    // ---- Pmax: generous, never truncates any printed cost table at q=Q ----
    ll maxCostAtQ = max({cD[Q], cE[Q]});
    for (auto& row : bg) maxCostAtQ = max(maxCostAtQ, row[Q]);
    ll Pmax = maxCostAtQ * 2 + 50;

    // ---- assemble & shuffle guild order (index != planted role) ----
    vector<vector<ll>> guilds;
    guilds.push_back(cD);
    guilds.push_back(cE);
    for (auto& row : bg) guilds.push_back(row);
    // deterministic shuffle via testlib rnd so guild identity isn't positional
    for (int i = (int)guilds.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(guilds[i], guilds[j]);
    }

    printf("%d %d %lld\n", Q, n, Pmax);
    for (int q = 0; q <= Q; q++) printf("%lld%c", V[q], q == Q ? '\n' : ' ');
    for (auto& row : guilds) {
        for (int q = 0; q <= Q; q++) printf("%lld%c", row[q], q == Q ? '\n' : ' ');
    }
    return 0;
}
