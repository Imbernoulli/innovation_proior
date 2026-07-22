#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Shatter the Syndicate".
// family: collective-influence-immunization    objective: MINIMIZE
//
// Input:  N M ELL K S ; then M lines "u v t" (1-based edge endpoints + robustness
//         threshold) ; then N immunization costs c_1..c_N ; then S contagion
//         strengths p_1..p_S.  Edge j is ACTIVE in scenario s iff t_j < p_s.
// Output: m (number of people immunized) then m DISTINCT ids in [1,N]. Feasible
//         iff the ids are distinct, in range, and sum of their costs <= K.
//
// Objective: for each scenario keep only its active edges among surviving
//   (non-immunized) people; L_s = size of the largest connected component.
//   Minimize F = sum_s L_s.  Baseline B = F for the empty immunization set
//   (do nothing), computed by the checker itself (B > 0 since N >= 1).
//   Score: sc = min(1000, 100*B/max(1,F)); ratio = sc/1000. Empty set -> ratio 0.1.
// -----------------------------------------------------------------------------

static int N, M, ELL, K, S;
static vector<int> eu, ev, et;
static vector<int> ps;

static vector<int> par, sz;
static int findp(int x){ while (par[x] != x){ par[x] = par[par[x]]; x = par[x]; } return x; }

static ll largestComp(int s, const vector<char>& removed){
    for (int i = 0; i < N; i++){ par[i] = i; sz[i] = removed[i] ? 0 : 1; }
    int p = ps[s];
    for (int j = 0; j < M; j++){
        if (et[j] >= p) continue;                 // edge not active in this scenario
        int a = eu[j], b = ev[j];
        if (removed[a] || removed[b]) continue;
        int ra = findp(a), rb = findp(b);
        if (ra != rb){ if (sz[ra] < sz[rb]) swap(ra, rb); par[rb] = ra; sz[ra] += sz[rb]; }
    }
    ll best = 0;
    for (int i = 0; i < N; i++) if (!removed[i] && findp(i) == i) best = max(best, (ll)sz[i]);
    return best;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    ELL = inf.readInt();
    K = inf.readInt();
    S = inf.readInt();
    (void)ELL;
    eu.resize(M); ev.resize(M); et.resize(M);
    for (int j = 0; j < M; j++){
        int u = inf.readInt(1, N, "u");
        int v = inf.readInt(1, N, "v");
        int t = inf.readInt(1, 999999, "t");
        eu[j] = u - 1; ev[j] = v - 1; et[j] = t;
    }
    vector<int> cost(N);
    for (int i = 0; i < N; i++) cost[i] = inf.readInt(1, 1000000000, "c");
    ps.resize(S);
    for (int s = 0; s < S; s++) ps[s] = inf.readInt(1, 999999, "p");

    par.assign(N, 0); sz.assign(N, 0);

    // ---- internal baseline B: immunize nobody ----
    vector<char> none(N, 0);
    ll B = 0;
    for (int s = 0; s < S; s++) B += largestComp(s, none);
    if (B <= 0) B = 1;

    // ---- read participant immunization set ----
    vector<char> removed(N, 0);
    int m = ouf.readInt(0, N, "m");
    vector<char> used(N, 0);
    ll spent = 0;
    for (int i = 0; i < m; i++){
        int x = ouf.readInt(1, N, "immunized_id");
        if (used[x - 1]) quitf(_wa, "person %d immunized more than once", x);
        used[x - 1] = 1;
        removed[x - 1] = 1;
        spent += cost[x - 1];
        if (spent > K) quitf(_wa, "immunization spend %lld exceeds budget K=%d", spent, K);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the immunization list");
    if (spent > K) quitf(_wa, "immunization spend %lld exceeds budget K=%d", spent, K);

    // ---- participant objective F ----
    ll F = 0;
    for (int s = 0; s < S; s++) F += largestComp(s, removed);
    if (F < 1) F = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld spent=%lld/%d Ratio: %.6f", F, B, spent, K, sc / 1000.0);
    return 0;
}
