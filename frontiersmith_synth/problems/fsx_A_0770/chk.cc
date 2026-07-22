#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Silencing a War-Drum Rehearsal Hall"
// family: room-mode-absorber-tuning
//
// Input:
//   W H                       room footprint (flavor)
//   sx sy                     drum (source) cell
//   NL ; then NL lines lx ly  listener cells
//   NM                        number of resonance channels (last one is the
//                             first-reflection / direct-path channel)
//   NM lines: Delta Gamma0 S L   (per channel: detuning, baseline damping,
//                                 drum coupling, listener coupling -- already
//                                 summed over all listeners)
//   Budget
//   M                         number of candidate wall panels
//   M lines: id x y cost alpha phi2_0 phi2_1 ... phi2_{NM-1}
//     (position is flavor only; cost/alpha are what you spend/choose;
//      phi2_k = this panel's coupling into channel k, i.e. how large that
//      channel's standing wave is AT this panel, scaled to [0, 1e6])
//
// Output: K then K distinct panel ids (1..M), sum cost <= Budget.
//
// Objective (MIN): for every channel k, extra damping from your chosen panels
//   Gamma_extra(k) = sum_{i in your set} alpha_i * phi2_i_k / 1000
//   Gamma(k) = Gamma0(k) + Gamma_extra(k)
//   E(k) = min(EK_CAP, S(k)^2 * SCALE2 / max(1, Delta(k)^2 + Gamma(k)^2))  (Lorentzian, capped)
//   F = sum_k E(k) * L(k)                                                (minimize)
//
// Baseline B: same formula with Gamma_extra(k) = 0 for all k (no panels
//   installed -- exactly what the trivial reference reproduces).
// Score: sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const ll SCALE2  = 1000000LL;
static const ll EK_CAP  = 1000000000LL;  // defensive cap, far above any realistic E(k)

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    inf.readLong(1LL, 10000LL, "W");
    inf.readLong(1LL, 10000LL, "H");
    inf.readLong(0LL, 10000LL, "sx");
    inf.readLong(0LL, 10000LL, "sy");
    int NL = inf.readInt(1, 1000, "NL");
    for (int j = 0; j < NL; j++){ inf.readLong(0LL, 10000LL, "lx"); inf.readLong(0LL, 10000LL, "ly"); }

    int NM = inf.readInt(1, 200, "NM");
    vector<ll> Delta(NM), Gamma0(NM), S(NM), L(NM);
    for (int k = 0; k < NM; k++){
        Delta[k]  = inf.readLong(0LL, 4000000LL, "Delta");
        Gamma0[k] = inf.readLong(1LL, 20000000LL, "Gamma0");
        S[k]      = inf.readLong(0LL, 300000LL, "S");
        L[k]      = inf.readLong(0LL, 5000000LL, "L");
    }

    ll Budget = inf.readLong(0LL, 2000000000LL, "Budget");
    int M = inf.readInt(1, 2000, "M");
    vector<ll> cost(M + 1), alpha(M + 1);
    vector<vector<ll>> phi2(M + 1, vector<ll>(NM));
    for (int i = 1; i <= M; i++){
        int id = inf.readInt(1, M, "id");
        if (id != i) quitf(_fail, "generator bug: candidate id out of order");
        inf.readLong(0LL, 10000LL, "x");
        inf.readLong(0LL, 10000LL, "y");
        cost[i]  = inf.readLong(1LL, 1000000LL, "cost");
        alpha[i] = inf.readLong(0LL, 1000LL, "alpha");
        for (int k = 0; k < NM; k++) phi2[i][k] = inf.readLong(0LL, 1000000LL, "phi2");
    }

    auto computeF = [&](vector<ll>& gammaExtra)->ll{
        ll F = 0;
        for (int k = 0; k < NM; k++){
            ll Gk = Gamma0[k] + gammaExtra[k];
            ll denom = Delta[k]*Delta[k] + Gk*Gk;
            if (denom < 1) denom = 1;
            ll Ek = (S[k]*S[k]) * SCALE2 / denom;
            if (Ek > EK_CAP) Ek = EK_CAP;
            F += Ek * L[k];
        }
        return F;
    };

    vector<ll> zeroExtra(NM, 0);
    ll B = computeF(zeroExtra);
    if (B <= 0) B = 1;

    int K = ouf.readInt(0, M, "K");
    vector<char> chosen(M + 1, 0);
    ll spent = 0;
    vector<ll> gammaExtra(NM, 0);
    for (int t = 0; t < K; t++){
        int id = ouf.readInt(1, M, "panel_id");
        if (chosen[id]) quitf(_wa, "panel %d selected more than once", id);
        chosen[id] = 1;
        spent += cost[id];
        if (spent > Budget) quitf(_wa, "budget exceeded: spent %lld > Budget %lld", spent, Budget);
        for (int k = 0; k < NM; k++) gammaExtra[k] += alpha[id] * phi2[id][k] / 1000;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = computeF(gammaExtra);
    if (F <= 0) F = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld K=%d spent=%lld/%lld Ratio: %.6f",
          F, B, K, spent, Budget, sc / 1000.0);
    return 0;
}
