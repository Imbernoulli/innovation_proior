// checker for "When Everyone Schedules Themselves"
// Simulates the deterministic best-response process (3 fixed adversarial starts) under
// (a) the participant's per-machine coordination rule and (b) the checker's own reference
// rule (longest-processing-time-first, LPT), and scores by comparing worst-case makespans.
#include "testlib.h"
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;

static int n, m;
static vector<vector<ll>> p;   // p[j][i], 0-indexed job j, machine i
static vector<ll> w;           // w[j]

static ll key_of(const vector<ll>&A, const vector<ll>&B, int i, int j){
    return A[i]*p[j][i] + B[i]*w[j];
}

// cost if job j is placed on machine i, given the OTHER occupants of i in `assign`
// (assign[j] itself is ignored -- we always evaluate "as if j sits on i").
static ll cost_join(int j, int i, const vector<int>& assign, const vector<ll>&A, const vector<ll>&B){
    ll kj = key_of(A,B,i,j);
    ll total = p[j][i];
    for(int jp=0; jp<n; jp++){
        if(jp==j) continue;
        if(assign[jp]==i){
            ll kjp = key_of(A,B,i,jp);
            if(kjp<kj || (kjp==kj && jp<j)) total += p[jp][i];
        }
    }
    return total;
}

static ll makespan(const vector<int>& assign){
    vector<ll> load(m,0);
    for(int j=0;j<n;j++) load[assign[j]] += p[j][assign[j]];
    ll mx=0; for(int i=0;i<m;i++) mx=max(mx,load[i]);
    return mx;
}

// deterministic best-response dynamics: sequential passes in job-index order,
// each job moves to whichever machine minimizes ITS OWN predicted completion time
// (strict improvement only); halts when a full pass makes no move or after a cap.
static vector<int> run_dynamics(vector<int> assign, const vector<ll>&A, const vector<ll>&B){
    const int PASS_CAP = 60;
    for(int pass=0; pass<PASS_CAP; pass++){
        bool moved=false;
        for(int j=0;j<n;j++){
            int c = assign[j];
            ll bestCost = cost_join(j,c,assign,A,B);
            int bestI = c;
            for(int i=0;i<m;i++){
                if(i==c) continue;
                ll ci = cost_join(j,i,assign,A,B);
                if(ci < bestCost){ bestCost=ci; bestI=i; }
            }
            if(bestI!=c){ assign[j]=bestI; moved=true; }
        }
        if(!moved) break;
    }
    return assign;
}

static vector<vector<int>> make_starts(){
    vector<ll> sum(m,0);
    for(int j=0;j<n;j++) for(int i=0;i<m;i++) sum[i]+=p[j][i];
    int slow=0, fast=0;
    for(int i=1;i<m;i++){
        if(sum[i]>sum[slow]) slow=i;
        if(sum[i]<sum[fast]) fast=i;
    }
    vector<int> s1(n,slow), s2(n,fast), s3(n);
    for(int j=0;j<n;j++) s3[j]=j%m;
    return {s1,s2,s3};
}

// worst-case equilibrium cost: max makespan reached over the 3 stress starts.
static ll worst_equilibrium(const vector<ll>&A, const vector<ll>&B){
    ll best=0;
    for(auto& s0 : make_starts()){
        auto fin = run_dynamics(s0, A, B);
        best = max(best, makespan(fin));
    }
    return best;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);
    n = inf.readInt(2, 60, "n");
    m = inf.readInt(2, 10, "m");
    p.assign(n, vector<ll>(m));
    w.assign(n, 0);
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++) p[j][i] = inf.readInt(1, 500, "p");
        w[j] = inf.readInt(1, 100, "w");
    }

    // ---- participant output: m lines of A_i B_i ----
    vector<ll> A(m), B(m);
    for(int i=0;i<m;i++){
        A[i] = ouf.readInt(-1000, 1000, "A_i");
        B[i] = ouf.readInt(-1000, 1000, "B_i");
    }
    if(!ouf.seekEof()) quitf(_wa, "trailing output after %d machine lines", m);

    // ---- internal reference baseline B: LPT (longest-processing-time-first) rule
    // on every machine (A_i=-1, B_i=0): a well-known, badly-suited-here fixed rule that
    // does not adapt per machine and ignores job weight entirely.
    vector<ll> Aref(m,-1), Bref(m,0);
    ll baseF = worst_equilibrium(Aref, Bref);
    if(baseF < 1) baseF = 1;

    ll F = worst_equilibrium(A, B);

    // Smooth open-ceiling score (minimization): matches baseline exactly -> 0.1;
    // asymptotes toward 1 as F shrinks well below the baseline (no hard saturation).
    double denom = (double)baseF + 9.0*(double)F;
    double ratio = (denom>0.0) ? (double)baseF/denom : 0.0;
    if(ratio<0.0) ratio=0.0; if(ratio>1.0) ratio=1.0;
    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, baseF, ratio);
}
