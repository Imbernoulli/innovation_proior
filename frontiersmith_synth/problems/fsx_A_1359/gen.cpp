// generator for "When Everyone Schedules Themselves"
// Builds instances on a size ladder (testId 1..10). Most cases are CLUSTER-TRAP instances:
// jobs are split into ncl groups, each group has a small "home" set of machines where it is
// fast and is slow everywhere else. This creates near-tied processing times among many jobs
// competing for the same machine, which is exactly the regime where a fixed textbook ordering
// rule (shortest-processing-time-first) either lets everyone self-select into an overcrowded
// machine, or (longest-first) misorders everyone -- while a rule that also weighs the job's
// importance, or simply refuses to let raw speed always win the front of the queue, settles
// into a better-balanced equilibrium. For every trap testId we generate several *candidate*
// instances internally and keep the one with the largest gap between the SPT-mechanism
// equilibrium and the best of a few alternative reference mechanisms -- i.e. the generator
// itself hunts for the trap, the same way the intended "strong" solver hunts for the rule.
#include "testlib.h"
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;

static int N, M;
static vector<vector<ll>> P;
static vector<ll> W;

static ll keyOf(const vector<ll>&A, const vector<ll>&B, int i, int j){
    return A[i]*P[j][i] + B[i]*W[j];
}
static ll costJoin(int j, int i, const vector<int>& assign, const vector<ll>&A, const vector<ll>&B){
    ll kj = keyOf(A,B,i,j);
    ll total = P[j][i];
    for(int jp=0; jp<N; jp++){
        if(jp==j) continue;
        if(assign[jp]==i){
            ll kjp = keyOf(A,B,i,jp);
            if(kjp<kj || (kjp==kj && jp<j)) total += P[jp][i];
        }
    }
    return total;
}
static ll makespanOf(const vector<int>& assign){
    vector<ll> load(M,0);
    for(int j=0;j<N;j++) load[assign[j]] += P[j][assign[j]];
    ll mx=0; for(int i=0;i<M;i++) mx=max(mx,load[i]);
    return mx;
}
static vector<int> runDyn(vector<int> assign, const vector<ll>&A, const vector<ll>&B, int passCap){
    for(int pass=0; pass<passCap; pass++){
        bool moved=false;
        for(int j=0;j<N;j++){
            int c=assign[j];
            ll bestCost = costJoin(j,c,assign,A,B);
            int bestI=c;
            for(int i=0;i<M;i++){
                if(i==c) continue;
                ll ci = costJoin(j,i,assign,A,B);
                if(ci<bestCost){ bestCost=ci; bestI=i; }
            }
            if(bestI!=c){ assign[j]=bestI; moved=true; }
        }
        if(!moved) break;
    }
    return assign;
}
static vector<vector<int>> starts3(){
    vector<ll> sum(M,0);
    for(int j=0;j<N;j++) for(int i=0;i<M;i++) sum[i]+=P[j][i];
    int slow=0, fast=0;
    for(int i=1;i<M;i++){ if(sum[i]>sum[slow]) slow=i; if(sum[i]<sum[fast]) fast=i; }
    vector<int> s1(N,slow), s2(N,fast), s3(N);
    for(int j=0;j<N;j++) s3[j]=j%M;
    return {s1,s2,s3};
}
static ll worstEq(const vector<ll>&A, const vector<ll>&B, int passCap){
    ll best=0;
    for(auto& s0 : starts3()){
        auto fin = runDyn(s0,A,B,passCap);
        best = max(best, makespanOf(fin));
    }
    return best;
}
// relative gap between the SPT mechanism and the best of several alternative reference rules
// (arbitrary/index order, weight-priority, SPT+weight-tiebreak, speed-rank-differentiated).
static double trapGap(int passCap){
    vector<ll> Aspt(M,1), Bspt(M,0);
    vector<ll> Aarb(M,0), Barb(M,0);
    vector<ll> Aw(M,0), Bw(M,-1);
    vector<ll> Amix(M,1), Bmix(M,-5);
    ll fSpt = worstEq(Aspt,Bspt,passCap);
    if(fSpt<=0) return 0.0;
    ll fArb = worstEq(Aarb,Barb,passCap);
    ll fW   = worstEq(Aw,Bw,passCap);
    ll fMix = worstEq(Amix,Bmix,passCap);
    // speed-rank-differentiated: below-median-load machines keep SPT, above-median add a
    // weight-priority tilt (discourages low-weight jobs from crowding the slower machines).
    vector<ll> sum(M,0);
    for(int j=0;j<N;j++) for(int i=0;i<M;i++) sum[i]+=P[j][i];
    vector<ll> sorted_sum=sum; sort(sorted_sum.begin(),sorted_sum.end());
    ll med = sorted_sum[M/2];
    vector<ll> Ad1(M,1), Bd1(M,0), Ad2(M,1), Bd2(M,0);
    for(int i=0;i<M;i++){
        if(sum[i]>med) Bd1[i]=-50; else Bd2[i]=-50;
    }
    ll fD1 = worstEq(Ad1,Bd1,passCap);
    ll fD2 = worstEq(Ad2,Bd2,passCap);
    ll best = min({fArb, fW, fMix, fD1, fD2});
    return (double)(fSpt-best)/(double)fSpt;
}

// build one cluster-trap candidate instance into P,W (N,M already set)
static void buildCluster(int ncl, int fastLo, int fastHi, int slowLo, int slowHi, double whiFrac){
    vector<int> cl(N);
    for(int j=0;j<N;j++) cl[j]=rnd.next(ncl);
    vector<int> base(ncl);
    for(int c=0;c<ncl;c++) base[c]=rnd.next(5,30);
    P.assign(N, vector<ll>(M));
    W.assign(N,0);
    for(int j=0;j<N;j++){
        int c=cl[j];
        for(int i=0;i<M;i++){
            if(i % ncl == c) P[j][i] = min(500, base[c] + rnd.next(0,5) + (fastHi-30>0? rnd.next(0, max(1,fastHi-fastLo)) : 0));
            else P[j][i] = rnd.next(slowLo, slowHi);
        }
        W[j] = (rnd.next(0.0,1.0) < whiFrac) ? rnd.next(70,100) : rnd.next(1,15);
    }
}
static void buildGeneric(){
    P.assign(N, vector<ll>(M));
    W.assign(N,0);
    for(int j=0;j<N;j++){
        for(int i=0;i<M;i++) P[j][i] = rnd.next(20,300);
        W[j] = rnd.next(1,100);
    }
}
// cluster-trap PLUS a needle: a couple of jobs with a dramatically cheap, easy-to-miss
// machine option buried among the cluster structure.
static void buildClusterNeedle(int ncl){
    buildCluster(ncl, 8, 20, 100, 300, 0.3);
    int needles = min(2,N);
    for(int j=0;j<needles;j++){ P[j][M-1] = rnd.next(2,5); W[j] = rnd.next(70,100); }
}

// search `trials` cluster candidates, keep the one with the largest trapGap
static void searchCluster(int ncl, int trials, int passCap, bool needle=false){
    vector<vector<ll>> bestP; vector<ll> bestW; double bestGap=-1.0;
    for(int t=0;t<trials;t++){
        if(needle) buildClusterNeedle(ncl);
        else buildCluster(ncl, 8, 20, 100, 300, 0.3 + 0.1*(t%3));
        double g = trapGap(passCap);
        if(g>bestGap){ bestGap=g; bestP=P; bestW=W; }
    }
    P=bestP; W=bestW;
}

static void printInstance(){
    printf("%d %d\n", N, M);
    for(int j=0;j<N;j++){
        for(int i=0;i<M;i++) printf("%lld ", P[j][i]);
        printf("%lld\n", W[j]);
    }
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // (n, m, ncl, kind, trials)  kind: 0=cluster-trap 1=generic 2=cluster+needle
    int n,mM,ncl,kind,trials;
    switch(testId){
        case 1:  n=8;  mM=2; ncl=2; kind=0; trials=70000; break;
        case 2:  n=10; mM=3; ncl=2; kind=0; trials=60000; break;
        case 3:  n=12; mM=3; ncl=2; kind=2; trials=48000; break; // cluster + needle
        case 4:  n=16; mM=3; ncl=2; kind=0; trials=36000; break;
        case 5:  n=16; mM=4; ncl=3; kind=0; trials=36000; break;
        case 6:  n=20; mM=4; ncl=3; kind=0; trials=26000; break;
        case 7:  n=22; mM=4; ncl=3; kind=0; trials=21000; break;
        case 8:  n=26; mM=4; ncl=3; kind=0; trials=15000; break;
        case 9:  n=32; mM=5; ncl=4; kind=0; trials=10000; break;
        default: n=60; mM=8; ncl=5; kind=0; trials=2600;  break; // testId 10: largest, fills envelope
    }
    N=n; M=mM;

    if(kind==0){
        int passCap = (n<=20)?26:(n<=32?20:14);
        searchCluster(ncl, trials, passCap);
    } else if(kind==1){
        buildGeneric();
    } else {
        int passCap = (n<=20)?26:(n<=32?20:16);
        searchCluster(ncl, trials, passCap, /*needle=*/true);
    }
    printInstance();
    return 0;
}
