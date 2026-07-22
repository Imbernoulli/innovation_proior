#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Route a Pipetting Protocol to Fill a Plate"
// family: serial-dilution-routing
//
// Input:  W M Vmax Vcap stepCost stockAccessCost maxOps
//         then W lines: c_i Vreq_i   (target concentration *1000, required volume)
//   N = W+M total wells (1..W target, W+1..N scratch/intermediate).
//
// Output (participant): K, then K lines "src dst vol"
//   src in {-1(stock),-2(diluent)} u [1,N]; dst in [1,N]; src!=dst; 1<=vol<=Vmax.
//   src, if a well, must currently hold >= vol of undrawn liquid.
//   dst's cumulative received volume (over the whole protocol) must not exceed Vcap.
//
// Evaluation: replay operations in order, tracking per-well (received, drawn, conc).
//   For i in 1..W:  acc_i = |conc_i - c_i/1000| + max(0,Vreq_i-recv_i)/Vreq_i
//   D = number of DISTINCT wells that ever receive liquid directly from stock (src=-1).
//   F (MIN) = 100*sum(acc_i) + stepCost*K + stockAccessCost*D
//
// Baseline B (checker-computed, do-nothing / empty protocol): every well stays at
//   conc 0, recv 0  ->  B = 100 * sum_i (c_i/1000 + 1).  Always positive.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    ll W = inf.readLong();
    ll M = inf.readLong();
    ll Vmax = inf.readLong();
    ll Vcap = inf.readLong();
    ll stepCost = inf.readLong();
    ll stockAccessCost = inf.readLong();
    ll maxOps = inf.readLong();
    ll N = W + M;

    vector<ll> c(W + 1), Vreq(W + 1);
    for (ll i = 1; i <= W; i++){
        c[i] = inf.readLong();
        Vreq[i] = inf.readLong();
    }

    // ---- internal baseline B = do-nothing (empty protocol) ----
    double B = 0.0;
    for (ll i = 1; i <= W; i++) B += (double)c[i] / 1000.0 + 1.0;
    B *= 100.0;
    if (B < 1e-9) B = 1e-9;

    // ---- read participant protocol (strict feasibility) ----
    ll K = ouf.readLong(0, maxOps, "K");

    vector<double> recv(N + 1, 0.0), drawn(N + 1, 0.0), conc(N + 1, 0.0);
    vector<char> stockTouched(N + 1, 0);
    ll D = 0;

    for (ll t = 0; t < K; t++){
        ll src = ouf.readLong(-2, N, "src");
        if (src == 0) quitf(_wa, "op %lld: src=0 is not a valid source", t + 1);
        ll dst = ouf.readLong(1, N, "dst");
        ll vol = ouf.readLong(1, Vmax, "vol");
        if (src == dst) quitf(_wa, "op %lld: src==dst (%lld)", t + 1, src);

        double srcConc;
        if (src == -1){
            srcConc = 1.0;
        } else if (src == -2){
            srcConc = 0.0;
        } else {
            double avail = recv[src] - drawn[src];
            if (avail < (double)vol - 1e-6)
                quitf(_wa, "op %lld: source well %lld has only %.3f undrawn, need %lld",
                      t + 1, src, avail, vol);
            srcConc = conc[src];
            drawn[src] += (double)vol;
        }

        double newRecv = recv[dst] + (double)vol;
        if (newRecv > (double)Vcap + 1e-6)
            quitf(_wa, "op %lld: dst well %lld capacity exceeded (%.3f > %lld)",
                  t + 1, dst, newRecv, Vcap);
        double newConc = (recv[dst] * conc[dst] + (double)vol * srcConc) / newRecv;
        if (!isfinite(newConc))
            quitf(_wa, "op %lld: non-finite concentration produced", t + 1);
        recv[dst] = newRecv;
        conc[dst] = newConc;

        if (src == -1){
            if (!stockTouched[dst]){ stockTouched[dst] = 1; D++; }
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing data after %lld operations", K);

    // ---- compute objective F ----
    double acc = 0.0;
    for (ll i = 1; i <= W; i++){
        double target = (double)c[i] / 1000.0;
        double dev = fabs(conc[i] - target);
        double shortfall = 0.0;
        if (recv[i] < (double)Vreq[i])
            shortfall = ((double)Vreq[i] - recv[i]) / (double)Vreq[i];
        acc += dev + shortfall;
    }
    double F = 100.0 * acc + (double)stepCost * (double)K + (double)stockAccessCost * (double)D;
    if (!isfinite(F)) quitf(_wa, "non-finite objective");
    if (F < 1.0) F = 1.0;

    double sc = min(1000.0, 100.0 * B / F);
    quitp(sc / 1000.0, "OK F=%.3f B=%.3f D=%lld K=%lld Ratio: %.6f",
          F, B, D, K, sc / 1000.0);
    return 0;
}
