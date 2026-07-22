#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Plates that split one cut two ways" (shared-kerf-pairing).
//
// Input:
//   N M
//   H_1 .. H_N
//   for g = 1..M:  "k_g L_g"  then  k_g member part indices (1-based, distinct)
//
// Output:
//   K
//   K lines: "g t p_1 .. p_t"   (g = 1-based common-line id, t>=2 distinct
//            members of line g who share that cut)
//
// baseline_i = sum over lines g with i in S_g of L_g/2 (mandatory solo-cut heat
// if i shares nothing). H_i - baseline_i = R_i, the EXTRA heat part i may spend
// on shared cuts (each shared line costs a participant its FULL L_g, i.e. an
// extra L_g/2 beyond the solo cost already folded into baseline_i).
//
// Objective F (max) = sum over activated lines g (t_g participants, t_g>=2) of
//   (t_g - 1) * L_g            -- physical cut-length saved by sharing.
//
// Internal baseline B: best ACHIEVABLE result using a single common line only
// (the "take the single best pair/cluster" reference) -- for each line g, count
// members that could individually afford it alone (R_i >= L_g/2); if that count
// m>=2 the line contributes (m-1)*L_g; B = max over g. This is exactly what the
// trivial reference solution reproduces -> ratio 0.1.
// Score: sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    vector<ll> H(N + 1);
    for (int i = 1; i <= N; i++) H[i] = inf.readLong();

    vector<vector<int>> mem(M + 1);
    vector<ll> L(M + 1);
    vector<ll> baseline(N + 1, 0);
    for (int g = 1; g <= M; g++){
        int k = inf.readInt();
        ll Lg = inf.readLong();
        L[g] = Lg;
        mem[g].resize(k);
        for (int j = 0; j < k; j++){
            int p = inf.readInt();
            mem[g][j] = p;
            baseline[p] += Lg / 2;
        }
    }
    vector<ll> Rextra(N + 1);
    for (int i = 1; i <= N; i++) Rextra[i] = H[i] - baseline[i];

    // ---- internal baseline B: best single common line ----
    ll B = 0;
    for (int g = 1; g <= M; g++){
        int m = 0;
        for (int p : mem[g]) if (Rextra[p] >= L[g] / 2) m++;
        if (m >= 2){ ll cand = (ll)(m - 1) * L[g]; if (cand > B) B = cand; }
    }
    if (B <= 0) B = 1; // generator guarantees a genuine positive baseline

    // ---- replay participant's shared-cut plan ----
    int K = ouf.readInt(0, M, "K");
    vector<char> groupUsed(M + 1, 0);
    vector<ll> spent(N + 1, 0);
    ll F = 0;
    for (int t = 0; t < K; t++){
        int g = ouf.readInt(1, M, "group_id");
        if (groupUsed[g]) quitf(_wa, "line %d activated more than once", g);
        groupUsed[g] = 1;
        int kg = (int)mem[g].size();
        int cnt = ouf.readInt(2, kg, "participant_count");
        vector<int> seen;
        vector<char> isMember(kg, 0);
        set<int> usedIdx; // guard against duplicate member positions being claimed twice
        for (int j = 0; j < cnt; j++){
            int p = ouf.readInt(1, N, "participant");
            // must be an actual member of line g
            int pos = -1;
            for (int q = 0; q < kg; q++) if (mem[g][q] == p){ pos = q; break; }
            if (pos < 0) quitf(_wa, "part %d is not on common line %d", p, g);
            if (!usedIdx.insert(pos).second) quitf(_wa, "part %d listed twice for line %d", p, g);
            spent[p] += L[g] / 2;
            if (spent[p] > Rextra[p])
                quitf(_wa, "part %d exceeds its heat budget (spent %lld > extra budget %lld)",
                      p, spent[p], Rextra[p]);
        }
        F += (ll)(cnt - 1) * L[g];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
