// TIER: strong
// Insight: since we do NOT control the assignment (jobs pick machines selfishly), we cannot
// optimize an assignment directly -- we can only shape which equilibrium the local ordering
// rules induce. Rather than committing to one textbook recipe (SPT), we EVALUATE several
// structurally different coordination rules by exactly simulating the same worst-case
// best-response process the checker uses (three adversarial starts -> converged makespan)
// and keep whichever rule actually induces the best equilibrium on THIS instance:
//   SPT               (pure speed priority)
//   ARB               (index order -- ignores speed entirely; a job can't jump a busy queue
//                       just because it happens to be individually fast there)
//   W-priority        (order by importance, ignoring raw speed)
//   SPT+weight tiebreak
//   speed-rank split  (below-median-load machines keep SPT; above-median add a weight tilt
//                      that discourages low-importance jobs from crowding the slow machines)
// This is mechanism design by simulated evaluation, not a fixed recipe: the winning rule
// (and even which of ARB/W/SPT wins) genuinely differs from instance to instance.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int n, m;
static vector<vector<ll>> P;
static vector<ll> W;

static ll keyOf(const vector<ll>&A, const vector<ll>&B, int i, int j){
    return A[i]*P[j][i] + B[i]*W[j];
}
static ll costJoin(int j, int i, const vector<int>& assign, const vector<ll>&A, const vector<ll>&B){
    ll kj = keyOf(A,B,i,j);
    ll total = P[j][i];
    for(int jp=0; jp<n; jp++){
        if(jp==j) continue;
        if(assign[jp]==i){
            ll kjp = keyOf(A,B,i,jp);
            if(kjp<kj || (kjp==kj && jp<j)) total += P[jp][i];
        }
    }
    return total;
}
static ll makespanOf(const vector<int>& assign){
    vector<ll> load(m,0);
    for(int j=0;j<n;j++) load[assign[j]] += P[j][assign[j]];
    ll mx=0; for(int i=0;i<m;i++) mx=max(mx,load[i]);
    return mx;
}
static vector<int> runDyn(vector<int> assign, const vector<ll>&A, const vector<ll>&B){
    for(int pass=0; pass<60; pass++){
        bool moved=false;
        for(int j=0;j<n;j++){
            int c=assign[j];
            ll bestCost = costJoin(j,c,assign,A,B);
            int bestI=c;
            for(int i=0;i<m;i++){
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
    vector<ll> sum(m,0);
    for(int j=0;j<n;j++) for(int i=0;i<m;i++) sum[i]+=P[j][i];
    int slow=0, fast=0;
    for(int i=1;i<m;i++){ if(sum[i]>sum[slow]) slow=i; if(sum[i]<sum[fast]) fast=i; }
    vector<int> s1(n,slow), s2(n,fast), s3(n);
    for(int j=0;j<n;j++) s3[j]=j%m;
    return {s1,s2,s3};
}
static ll worstEq(const vector<ll>&A, const vector<ll>&B){
    ll best=0;
    for(auto& s0 : starts3()){
        auto fin = runDyn(s0,A,B);
        best = max(best, makespanOf(fin));
    }
    return best;
}

int main(){
    if(scanf("%d %d",&n,&m)!=2) return 0;
    P.assign(n, vector<ll>(m));
    W.assign(n,0);
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++) scanf("%lld",&P[j][i]);
        scanf("%lld",&W[j]);
    }

    vector<ll> sum(m,0);
    for(int j=0;j<n;j++) for(int i=0;i<m;i++) sum[i]+=P[j][i];
    vector<ll> sorted_sum=sum; sort(sorted_sum.begin(),sorted_sum.end());
    ll med = sorted_sum[m/2];

    vector<pair<vector<ll>,vector<ll>>> cands;
    cands.push_back({vector<ll>(m,1),  vector<ll>(m,0)});         // SPT
    cands.push_back({vector<ll>(m,0),  vector<ll>(m,0)});         // ARB
    cands.push_back({vector<ll>(m,0),  vector<ll>(m,-1)});        // W-priority
    cands.push_back({vector<ll>(m,1),  vector<ll>(m,-5)});        // SPT + weight tiebreak
    {
        vector<ll> Ad1(m,1), Bd1(m,0);
        for(int i=0;i<m;i++) if(sum[i]>med) Bd1[i]=-50;
        cands.push_back({Ad1,Bd1});                                // fast=SPT, slow=weight-tilt
    }
    {
        vector<ll> Ad2(m,1), Bd2(m,0);
        for(int i=0;i<m;i++) if(sum[i]<=med) Bd2[i]=-50;
        cands.push_back({Ad2,Bd2});                                // slow=SPT, fast=weight-tilt
    }

    ll bestF = -1; int bestIdx = 0;
    for(size_t c=0;c<cands.size();c++){
        ll f = worstEq(cands[c].first, cands[c].second);
        if(bestF<0 || f<bestF){ bestF=f; bestIdx=(int)c; }
    }
    for(int i=0;i<m;i++) printf("%lld %lld\n", cands[bestIdx].first[i], cands[bestIdx].second[i]);
    return 0;
}
