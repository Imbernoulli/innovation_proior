#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Factory heat-exchanger web"  (family: pinch-cascade-matching)
//
// Input:  NH NC dTmin ; NH hot lines (THs THt CP, THs>THt) ; NC cold lines
//         (TCs TCt CP, TCs<TCt).  CP = heat-capacity flow (heat per degree).
// Hot stream capacity  capH = CP*(THs-THt) ; cold capacity capC = CP*(TCt-TCs).
//
// Output: M ; then M lines "h c Q" : exchanger transferring Q>0 units of heat from
//         hot h to cold c.  Matches are applied IN THE LISTED ORDER; each stream
//         keeps a running exchanged-heat total (q). For a match the ENTERING temps
//         are  Hin = THs - qh/CPh ,  Cin = TCs + qc/CPc ; the EXITING temps are
//         Hout = THs-(qh+Q)/CPh ,  Cout = TCs+(qc+Q)/CPc . Counter-current feasibility
//         requires  Hin-Cout >= dTmin  AND  Hout-Cin >= dTmin. Also qh+Q<=capH,
//         qc+Q<=capC (no stream may be pushed past its target). A stream may appear
//         in many matches (its duty is split). All temps/CP integers, so the two
//         conditions are checked exactly after multiplying by CPh*CPc:
//            A) (THs-TCs-dTmin)*CPh*CPc - qh*CPc - (qc+Q)*CPh >= 0
//            B) (THs-TCs-dTmin)*CPh*CPc - (qh+Q)*CPc - qc*CPh >= 0
//
// Objective (MIN): external utility  F = sum(capH-qh) + sum(capC-qc)   (all heat
//   not recovered must be bought as cold/hot utility). Recovered heat R = sum Q,
//   and F = B - 2R where B = sum capH + sum capC (the do-nothing utility).
// Baseline B = do-nothing (no exchangers). Trivial reference "M=0" -> F=B -> 0.1.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000 (cap 1.0 = 10x).
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int NH = inf.readInt();
    int NC = inf.readInt();
    ll dTmin = inf.readLong();
    vector<ll> Hs(NH+1), Ht(NH+1), CPh(NH+1), capH(NH+1), qh(NH+1, 0);
    vector<ll> Cs(NC+1), Ct(NC+1), CPc(NC+1), capC(NC+1), qc(NC+1, 0);
    for (int i = 1; i <= NH; i++){
        Hs[i]=inf.readLong(); Ht[i]=inf.readLong(); CPh[i]=inf.readLong();
        capH[i]=CPh[i]*(Hs[i]-Ht[i]);
    }
    for (int j = 1; j <= NC; j++){
        Cs[j]=inf.readLong(); Ct[j]=inf.readLong(); CPc[j]=inf.readLong();
        capC[j]=CPc[j]*(Ct[j]-Cs[j]);
    }
    ll B = 0;
    for (int i=1;i<=NH;i++) B += capH[i];
    for (int j=1;j<=NC;j++) B += capC[j];
    if (B <= 0) B = 1;

    int M = ouf.readInt(0, 4000000, "M");
    for (int e = 0; e < M; e++){
        int h = ouf.readInt(1, NH, "h");
        int c = ouf.readInt(1, NC, "c");
        ll Q = ouf.readLong(1, (ll)4e18, "Q");
        if (qh[h] + Q > capH[h])
            quitf(_wa, "match %d: hot %d over-cooled (qh=%lld+Q=%lld > capH=%lld)", e, h, qh[h], Q, capH[h]);
        if (qc[c] + Q > capC[c])
            quitf(_wa, "match %d: cold %d over-heated (qc=%lld+Q=%lld > capC=%lld)", e, c, qc[c], Q, capC[c]);
        ll K = (Hs[h] - Cs[c] - dTmin) * CPh[h] * CPc[c];
        ll condA = K - qh[h]*CPc[c] - (qc[c]+Q)*CPh[h];   // Hin - Cout >= dTmin
        ll condB = K - (qh[h]+Q)*CPc[c] - qc[c]*CPh[h];   // Hout - Cin >= dTmin
        if (condA < 0 || condB < 0)
            quitf(_wa, "match %d (hot %d, cold %d, Q=%lld): dTmin=%lld violated (A=%lld B=%lld)",
                  e, h, c, Q, dTmin, condA, condB);
        qh[h] += Q; qc[c] += Q;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the match list");

    ll F = 0;
    for (int i=1;i<=NH;i++) F += capH[i]-qh[i];
    for (int j=1;j<=NC;j++) F += capC[j]-qc[j];

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc/1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc/1000.0);
    return 0;
}
