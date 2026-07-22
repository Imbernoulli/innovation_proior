#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Mining the Cannibal River" (cannibal-cascade-heat-mining).
//
// Input:  T K Tsink ; then T lines  q_i theta_i eta_i cap_i.
//
// Output (participant): m ; then m lines  p x  (install a converter at segment
//   p extracting heat x there; 0<=m<=K, all p distinct, 1<=p<=T).
//
// Feasibility (checked in increasing p order, chain mixing accounted for):
//   0 <= x_p <= min(cap_p, F_p * (Tpre_p - Tsink))
//   where F_p = cumulative flow reaching p, Tpre_p = mixed inbound temperature
//   (reflecting every upstream extraction already applied).
//
// Objective (MAX): for each installed p with Tpost_p = Tpre_p - x_p/F_p,
//   E_p = eta_p * F_p * [ (Tpre_p-Tpost_p) - Tsink*ln(Tpre_p/Tpost_p) ]
//   F = sum E_p.
//
// Baseline B (checker-computed, do-nothing-clever reference): under the
//   NATURAL (no-extraction) mixing profile, the best SINGLE segment's own
//   capped/Tsink-limited electricity output. B is always > 0 since every
//   theta_i > Tsink and every cap_i >= 1.
//
// Score (max): sc = min(1000, 100*F/max(eps,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const double EPS = 1e-6;

static double energyOf(double e, double F, double Tpre, double Tpost, double Tsink) {
    if (Tpost > Tpre) Tpost = Tpre;
    if (Tpost < Tsink) Tpost = Tsink;
    if (Tpost >= Tpre) return 0.0;
    return e * F * ((Tpre - Tpost) - Tsink * log(Tpre / Tpost));
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int T = inf.readInt();
    int K = inf.readInt();
    int Tsink = inf.readInt();
    vector<ll> q(T + 1), theta(T + 1), eta(T + 1), cap(T + 1);
    for (int i = 1; i <= T; i++) {
        q[i] = inf.readLong();
        theta[i] = inf.readLong();
        eta[i] = inf.readLong();
        cap[i] = inf.readLong();
    }

    // ---- natural (no-extraction) profile: cumulative flow + mixed temperature ----
    vector<double> natPre(T + 1), natFlow(T + 1);
    {
        double F = 0.0, Tcur = 0.0;
        for (int i = 1; i <= T; i++) {
            double Fnew = F + (double)q[i];
            double Tpre = (F == 0.0) ? (double)theta[i] : (F * Tcur + (double)q[i] * theta[i]) / Fnew;
            F = Fnew;
            natPre[i] = Tpre; natFlow[i] = F;
            Tcur = Tpre;
        }
    }

    // ---- internal baseline B: best single capped/Tsink-limited segment ----
    double B = -1.0;
    for (int i = 1; i <= T; i++) {
        double e = eta[i] / 1000.0;
        double feasMax = natFlow[i] * (natPre[i] - Tsink);
        double xmax = min((double)cap[i], feasMax);
        if (xmax < 0) xmax = 0;
        double Tpost = natPre[i] - xmax / natFlow[i];
        double v = energyOf(e, natFlow[i], natPre[i], Tpost, Tsink);
        if (v > B) B = v;
    }
    if (B < 1e-9) B = 1e-9;  // defensive; natPre[i] > Tsink always so B>0 in practice

    // ---- read participant output (strict bounded feasibility) ----
    int m = ouf.readInt(0, K, "m");
    vector<pair<int,ll>> raw(m);
    vector<char> seen(T + 1, 0);
    for (int j = 0; j < m; j++) {
        int p = ouf.readInt(1, T, "p");
        if (seen[p]) quitf(_wa, "duplicate position %d", p);
        seen[p] = 1;
        ll x = ouf.readLong(0, (ll)2e9, "x");
        raw[j] = {p, x};
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after last (p,x) pair");

    sort(raw.begin(), raw.end());
    vector<ll> instX(T + 1, -1);
    for (auto& pr : raw) instX[pr.first] = pr.second;

    // ---- forward simulation of the participant's construction ----
    double F = 0.0, Tcur = 0.0, Etot = 0.0;
    for (int i = 1; i <= T; i++) {
        double Fnew = F + (double)q[i];
        double Tpre = (F == 0.0) ? (double)theta[i] : (F * Tcur + (double)q[i] * theta[i]) / Fnew;
        F = Fnew;
        if (instX[i] >= 0) {
            ll x = instX[i];
            double feasMax = F * (Tpre - Tsink);
            double bound = min((double)cap[i], feasMax);
            if ((double)x > bound + max(EPS, bound * 1e-9) + 1e-6)
                quitf(_wa, "infeasible extraction at segment %d: x=%lld > bound=%.6f (cap=%lld, feasMax=%.6f)",
                      i, x, bound, cap[i], feasMax);
            double Tpost = Tpre - (double)x / F;
            if (Tpost < Tsink - 1e-6)
                quitf(_wa, "segment %d cooled below Tsink: Tpost=%.6f Tsink=%d", i, Tpost, Tsink);
            if (Tpost > Tsink) { /* fine */ } else { Tpost = Tsink; }
            double e = eta[i] / 1000.0;
            Etot += energyOf(e, F, Tpre, Tpost, Tsink);
            Tcur = Tpost;
        } else {
            Tcur = Tpre;
        }
    }

    double sc = min(1000.0, 100.0 * Etot / max(1e-9, B));
    if (!isfinite(sc) || sc < 0) sc = 0.0;
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", Etot, B, sc / 1000.0);
    return 0;
}
