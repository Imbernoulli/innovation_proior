#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Accordion Silhouette Scheduling".
//
// Input:  n T ; then n-1 tokens req_i ("M"/"V") ; then n-1 positive weights w_i.
// Output: k ; then k lines "c d" -- c = crease committed at that (1-indexed by
//   print order) step, d = "M"/"V". Crease i is EXPOSED at commit time iff
//   i==1, i==n-1, or a neighboring crease was already committed earlier in the
//   participant's own list.
//
// Objective (MAX): step j commits crease c with direction d ->
//   contribution = w_c * j * (2 if d==req_c else 1)     [x2 scale, /2 later]
//   RAW = sum of contributions. Mc,Vc = # committed M / V.
//   F = RAW * (2 if Mc-Vc==T else 1)                     [x2 scale, /2 later]
// (The x2/x2 doubling keeps everything integer; it's a constant global factor
//  applied identically to F and the baseline B below, so it cancels exactly in
//  the ratio F/B and matches the statement's 1.0/0.5 semantics exactly.)
//
// Baseline B (checker-computed): commit every crease in plain ascending index
//   order 1,2,...,n-1 (always exposed, since crease i's left neighbor i-1 was
//   just committed), always folding Mountain (never reading req_i, never
//   reasoning about T). This is exactly what the trivial reference reproduces
//   (-> ratio 0.1).
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    ll n = inf.readLong((ll)3, (ll)200000, "n");
    ll km1 = n - 1;
    ll T = inf.readLong((ll)(-km1), (ll)km1, "T");

    vector<char> req(km1 + 1, 'M');
    for (ll i = 1; i <= km1; i++){
        string tok = inf.readToken();
        req[i] = tok[0]; // input is generator-trusted: "M" or "V"
    }
    vector<ll> w(km1 + 1, 0);
    for (ll i = 1; i <= km1; i++) w[i] = inf.readLong((ll)1, (ll)50000, "w");

    // ---- internal baseline B: ascending commit order 1..km1, always "M" ----
    // (naive: never reads req_i, never reasons about the parity target T)
    ll rawB = 0, gapB = 0;
    for (ll i = 1; i <= km1; i++){
        ll match = (req[i] == 'M') ? 2 : 1;
        rawB += w[i] * i * match;
        gapB += 1; // always "M"
    }
    ll gfB = (gapB == T) ? 2 : 1;
    ll B = rawB * gfB;
    if (B <= 0) B = 1;

    // ---- replay the participant's commit sequence ----
    ll k = ouf.readLong((ll)0, km1, "k");
    vector<char> folded(km1 + 2, 0);
    ll raw = 0, Mc = 0, Vc = 0;

    for (ll t = 1; t <= k; t++){
        ll c = ouf.readLong((ll)1, km1, "crease");
        if (folded[c]) quitf(_wa, "crease %lld committed twice", c);
        bool exposed = (c == 1) || (c == km1) ||
                       (folded[c - 1]) || (folded[c + 1]);
        if (!exposed)
            quitf(_wa, "crease %lld not exposed at step %lld (neighbors uncommitted)", c, t);
        string dtok = ouf.readToken();
        if (dtok != "M" && dtok != "V")
            quitf(_wa, "direction token must be 'M' or 'V', got '%s'", dtok.c_str());
        char d = dtok[0];
        folded[c] = 1;
        ll match = (d == req[c]) ? 2 : 1;
        raw += w[c] * t * match;
        if (d == 'M') Mc++; else Vc++;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after %lld commits", k);

    ll gap = Mc - Vc;
    ll gf = (gap == T) ? 2 : 1;
    ll F = raw * gf;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld k=%lld Mc=%lld Vc=%lld gap=%lld T=%lld Ratio: %.6f",
          F, B, k, Mc, Vc, gap, T, sc / 1000.0);
    return 0;
}
