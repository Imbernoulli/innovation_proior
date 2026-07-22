#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Checker for "Analyzer Lineup: Tips and Recalibrations".
// Input : n K1 K2 GAP_COEF DRIFT_COEF BOOST  then n lines conc_i is_stat_i deadline_i
// Output: 3 lines of n ints -- p (permutation), tip (0/1), recal (0/1).
// F = sum_j carry_j + drift_j  (see statement.txt for the exact recurrence).
// B = same objective for the fixed reference plan p_j=j, no resets (always
//     feasible since the generator guarantees deadline_i >= i).
// ratio = min(1000, 100*B/max(1,F)) / 1000.

static const ll CAP = 60; // drift plateau: usage-count effect stops growing past CAP

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt(2, 4000, "n");
    int K1 = inf.readInt(0, n, "K1");
    int K2 = inf.readInt(0, n, "K2");
    ll GAP_COEF = inf.readInt(1, 10, "GAP_COEF");
    ll DRIFT_COEF = inf.readInt(1, 10, "DRIFT_COEF");
    ll BOOST = inf.readInt(0, 100, "BOOST");

    vector<ll> conc(n + 1);
    vector<int> isStat(n + 1, 0);
    vector<int> deadline(n + 1, n);
    for (int i = 1; i <= n; i++) {
        conc[i] = inf.readInt(1, 200000, "conc");
        isStat[i] = inf.readInt(0, 1, "is_stat");
        deadline[i] = inf.readInt(1, n, "deadline");
        if (isStat[i] && deadline[i] < i) quitf(_fail, "generator produced deadline < index");
    }

    // ---- participant output ----------------------------------------------
    vector<int> p(n + 1), pos(n + 1, 0);
    vector<char> seen(n + 1, 0);
    for (int j = 1; j <= n; j++) {
        int v = ouf.readInt(1, n, "p_j");
        if (seen[v]) quitf(_wa, "sample id repeated in run order");
        seen[v] = 1;
        p[j] = v;
        pos[v] = j;
    }

    vector<int> tip(n + 1, 0);
    ll tipSum = 0;
    for (int j = 1; j <= n; j++) {
        tip[j] = ouf.readInt(0, 1, "tip_j");
        tipSum += tip[j];
    }
    if (tipSum > K1) quitf(_wa, "tip-change budget exceeded");

    vector<int> recal(n + 1, 0);
    ll recalSum = 0;
    for (int j = 1; j <= n; j++) {
        recal[j] = ouf.readInt(0, 1, "recal_j");
        recalSum += recal[j];
    }
    if (recalSum > K2) quitf(_wa, "recalibration budget exceeded");

    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after output");

    for (int i = 1; i <= n; i++) {
        if (isStat[i] && pos[i] > deadline[i])
            quitf(_wa, "stat sample %d placed at position %d, deadline %d", i, pos[i], deadline[i]);
    }

    // ---- objective for participant's plan ----------------------------------
    auto evalPlan = [&](const vector<int>& ord, const vector<int>& tipF, const vector<int>& recalF) -> ll {
        ll F = 0;
        ll prevConc = 0, cnt = 0;
        for (int j = 1; j <= n; j++) {
            ll c = conc[ord[j]];
            ll gap = (j == 1) ? 0 : max((ll)0, prevConc - c);
            ll carry = tipF[j] ? 0 : GAP_COEF * gap;
            cnt = recalF[j] ? 1 : cnt + 1;
            ll e = min(cnt, CAP);
            bool boosted = (j > 1) && (gap > 0) && !tipF[j];
            ll drift = DRIFT_COEF * e * e * (boosted ? (1 + BOOST) : 1);
            F += carry + drift;
            prevConc = c;
        }
        return F;
    };

    ll F = evalPlan(p, tip, recal);

    // ---- internal baseline: identity order, no resets at all ---------------
    vector<int> idOrd(n + 1), zeroF(n + 1, 0);
    for (int i = 1; i <= n; i++) idOrd[i] = i;
    ll B = evalPlan(idOrd, zeroF, zeroF);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
