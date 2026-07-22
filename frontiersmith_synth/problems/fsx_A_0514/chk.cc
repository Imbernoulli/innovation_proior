#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for "Reinforcing Footbridges across a Branching Canyon System" (MAXIMIZE).
// Participant lists new footbridges (distinct unordered pairs, not already present); building
// bridge (u,v) costs the ORIGINAL hop-distance d_G(u,v); total cost <= B. Objective F = algebraic
// connectivity lambda_2 (2nd-smallest Laplacian eigenvalue) of the augmented graph, computed by a
// deterministic Lanczos iteration + Sturm bisection. Internal baseline Bb = lambda_2 of the
// ORIGINAL graph (do-nothing).  ratio = min(1, 0.1 * F / Bb)  ->  do-nothing scores 0.1.

static int N;

// smallest positive Laplacian eigenvalue (lambda_2) via Lanczos on the mean-zero subspace.
static double lanczos_lam2(const vector<vector<int>>& adj){
    int n=N;
    if(n<=1) return 0.0;
    int m=min(n-1,220);
    vector<double> q(n), qprev(n,0.0), w(n);
    { // deterministic mean-zero unit start vector
        unsigned long long s=88172645463325252ULL; double mean=0;
        for(int i=0;i<n;i++){ s^=s<<13; s^=s>>7; s^=s<<17;
            q[i]=((double)(s>>11)*(1.0/9007199254740992.0))-0.5; mean+=q[i]; }
        mean/=n; double nn=0; for(int i=0;i<n;i++){ q[i]-=mean; nn+=q[i]*q[i]; } nn=sqrt(nn);
        if(nn<1e-300) nn=1; for(int i=0;i<n;i++) q[i]/=nn;
    }
    vector<vector<double>> Q; Q.reserve(m);
    vector<double> alpha, off; double beta=0.0; int k=0;
    for(int j=0;j<m;j++){
        for(int i=0;i<n;i++){ double s=0; for(int x:adj[i]) s+=q[x];
            w[i]=(double)adj[i].size()*q[i]-s; }
        if(j>0) for(int i=0;i<n;i++) w[i]-=beta*qprev[i];
        double a=0; for(int i=0;i<n;i++) a+=w[i]*q[i];
        for(int i=0;i<n;i++) w[i]-=a*q[i];
        for(int pass=0;pass<2;pass++){
            double mean=0; for(int i=0;i<n;i++) mean+=w[i]; mean/=n;
            for(int i=0;i<n;i++) w[i]-=mean;
            for(auto &qq:Q){ double d=0; for(int i=0;i<n;i++) d+=w[i]*qq[i];
                for(int i=0;i<n;i++) w[i]-=d*qq[i]; }
            double d=0; for(int i=0;i<n;i++) d+=w[i]*q[i];
            for(int i=0;i<n;i++) w[i]-=d*q[i];
        }
        alpha.push_back(a); Q.push_back(q); k=j+1;
        double b=0; for(int i=0;i<n;i++) b+=w[i]*w[i]; b=sqrt(b);
        if(b<1e-10) break;
        if(j+1>=m) break;
        off.push_back(b); qprev=q; beta=b;
        for(int i=0;i<n;i++) q[i]=w[i]/b;
    }
    // smallest eigenvalue of the tridiagonal (alpha[0..k-1], off[0..k-2]) via Sturm bisection
    auto count_lt=[&](double x)->int{
        int cnt=0; double d=alpha[0]-x; if(d<0) cnt++;
        for(int i=1;i<k;i++){ if(fabs(d)<1e-300) d=(d<0?-1e-300:1e-300);
            d=(alpha[i]-x)-off[i-1]*off[i-1]/d; if(d<0) cnt++; }
        return cnt;
    };
    double glo=1e300, ghi=-1e300;
    for(int i=0;i<k;i++){
        double r=(i>0?fabs(off[i-1]):0)+(i<k-1?fabs(off[i]):0);
        glo=min(glo,alpha[i]-r); ghi=max(ghi,alpha[i]+r);
    }
    double lo=glo-1.0, hi=ghi+1.0;
    for(int it=0;it<200 && hi-lo>1e-13*max(1.0,fabs(hi));it++){
        double mid=0.5*(lo+hi);
        if(count_lt(mid)>=1) hi=mid; else lo=mid;
    }
    return hi;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);
    int n=inf.readInt(); int m=inf.readInt(); long long B=inf.readLong();
    N=n;
    vector<vector<int>> adj(n);
    unordered_set<long long> eset; eset.reserve(2*m+16);
    auto key=[&](int a,int b)->long long{ if(a>b) swap(a,b); return (long long)a*(n+1)+b; };
    for(int i=0;i<m;i++){
        int u=inf.readInt(1,n)-1, v=inf.readInt(1,n)-1;
        adj[u].push_back(v); adj[v].push_back(u); eset.insert(key(u,v));
    }

    // ---- read participant new bridges ----
    long long kmax=min((long long)5*B+16, (long long)2000000);
    int K=ouf.readInt(0,(int)kmax,"K");
    vector<pair<int,int>> newe; newe.reserve(K);
    unordered_set<long long> nset; nset.reserve(2*K+16);
    for(int i=0;i<K;i++){
        int u=ouf.readInt(1,n,"u")-1, v=ouf.readInt(1,n,"v")-1;
        if(u==v) quitf(_wa,"bridge %d connects a platform to itself (u=v=%d)",i+1,u+1);
        long long kk=key(u,v);
        if(eset.count(kk)) quitf(_wa,"bridge %d = (%d,%d) already exists in the network",i+1,u+1,v+1);
        if(nset.count(kk)) quitf(_wa,"bridge %d = (%d,%d) is a duplicate new bridge",i+1,u+1,v+1);
        nset.insert(kk); newe.push_back({u,v});
    }
    if(!ouf.seekEof()) quitf(_wa,"trailing output after the bridge list");

    // ---- cost = original hop distance (BFS, cached per source) ----
    unordered_map<int,vector<int>> distc;
    auto bfs=[&](int s)->vector<int>&{
        auto it=distc.find(s);
        if(it!=distc.end()) return it->second;
        vector<int> d(n,-1); d[s]=0; queue<int>q; q.push(s);
        while(!q.empty()){ int x=q.front();q.pop();
            for(int y:adj[x]) if(d[y]<0){ d[y]=d[x]+1; q.push(y); } }
        return distc.emplace(s,move(d)).first->second;
    };
    long long cost=0;
    for(auto &pr:newe){
        int u=pr.first, v=pr.second;
        int s=(adj[u].size()<=adj[v].size())?u:v, o=(s==u?v:u);
        int d=bfs(s)[o];
        if(d<0) quitf(_wa,"bridge (%d,%d) endpoints are unreachable in the original network",u+1,v+1);
        cost+=d;
    }
    if(cost>B) quitf(_wa,"total build cost %lld exceeds budget B=%lld",cost,B);

    // ---- augmented graph & objective ----
    vector<vector<int>> aug=adj;
    for(auto &pr:newe){ aug[pr.first].push_back(pr.second); aug[pr.second].push_back(pr.first); }

    double Bb=lanczos_lam2(adj);   // baseline: do-nothing algebraic connectivity
    double F =lanczos_lam2(aug);   // participant's algebraic connectivity
    if(!(Bb>0.0)) Bb=1e-9;
    if(!isfinite(F) || F<0) F=0.0;

    double ratio=0.1*F/Bb;
    if(ratio<0) ratio=0;
    if(ratio>1.0) ratio=1.0;
    quitp(ratio,"OK F=%.8f Bb=%.8f cost=%lld/%lld Ratio: %.6f",F,Bb,cost,B,ratio);
    return 0;
}
