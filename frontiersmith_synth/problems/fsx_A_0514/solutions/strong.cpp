// TIER: strong
// Multi-scale, cost-aware Fiedler augmentation. The insight: an edge's first-order gain in
// lambda_2 is ~(f_u - f_v)^2 while its price is the hop distance, so the winning move maximizes
// (df)^2 / cost -- and a bridge planted right at a GATEWAY carries the FULL Fiedler jump across
// the bottleneck at tiny distance, dominating greedy's deep, expensive extreme-node bridges.
// After each batch we RECOMPUTE the Fiedler vector: repairing the top cut makes the bottleneck
// migrate to a deeper scale, and the successive Fiedler vectors chase it there. Budget is thus
// spread across scales via many cheap mid-range bridges instead of dumped on one visible cut.
#include <bits/stdc++.h>
using namespace std;

int N;
static vector<double> tri_eigvec_small(const vector<double>&al,const vector<double>&of,double mu){
    int k=al.size(); vector<double> y(k,1.0);
    for(int iter=0;iter<3;iter++){
        vector<double> c(k,0.0), d(k,0.0);
        double piv=al[0]-mu; if(fabs(piv)<1e-30) piv=(piv<0?-1e-30:1e-30);
        c[0]=(k>1?of[0]:0.0)/piv; d[0]=y[0]/piv;
        for(int i=1;i<k;i++){ double den=(al[i]-mu)-of[i-1]*c[i-1];
            if(fabs(den)<1e-30) den=(den<0?-1e-30:1e-30);
            c[i]=(i<k-1?of[i]:0.0)/den; d[i]=(y[i]-of[i-1]*d[i-1])/den; }
        vector<double> x(k); x[k-1]=d[k-1];
        for(int i=k-2;i>=0;i--) x[i]=d[i]-c[i]*x[i+1];
        double nn=0; for(double v:x) nn+=v*v; nn=sqrt(nn); if(nn<1e-300)nn=1;
        for(int i=0;i<k;i++) y[i]=x[i]/nn;
    }
    return y;
}
vector<double> fiedler(const vector<vector<int>>& adj,int msteps){
    int n=N; vector<double> f(n,0.0); if(n<=1) return f;
    int m=min(n-1,msteps);
    vector<double> q(n),qprev(n,0.0),w(n);
    { unsigned long long s=88172645463325252ULL; double mean=0;
      for(int i=0;i<n;i++){ s^=s<<13; s^=s>>7; s^=s<<17;
          q[i]=((double)(s>>11)*(1.0/9007199254740992.0))-0.5; mean+=q[i]; }
      mean/=n; double nn=0; for(int i=0;i<n;i++){ q[i]-=mean; nn+=q[i]*q[i]; } nn=sqrt(nn);
      if(nn<1e-300)nn=1; for(int i=0;i<n;i++) q[i]/=nn; }
    vector<vector<double>> Q; vector<double> al,of; double beta=0; int k=0;
    for(int j=0;j<m;j++){
        for(int i=0;i<n;i++){ double s=0; for(int x:adj[i]) s+=q[x];
            w[i]=(double)adj[i].size()*q[i]-s; }
        if(j>0) for(int i=0;i<n;i++) w[i]-=beta*qprev[i];
        double a=0; for(int i=0;i<n;i++) a+=w[i]*q[i];
        for(int i=0;i<n;i++) w[i]-=a*q[i];
        for(int pass=0;pass<2;pass++){
            double mn=0; for(int i=0;i<n;i++) mn+=w[i]; mn/=n; for(int i=0;i<n;i++) w[i]-=mn;
            for(auto&qq:Q){ double d=0; for(int i=0;i<n;i++) d+=w[i]*qq[i];
                for(int i=0;i<n;i++) w[i]-=d*qq[i]; }
            double d=0; for(int i=0;i<n;i++) d+=w[i]*q[i]; for(int i=0;i<n;i++) w[i]-=d*q[i];
        }
        al.push_back(a); Q.push_back(q); k=j+1;
        double b=0; for(int i=0;i<n;i++) b+=w[i]*w[i]; b=sqrt(b);
        if(b<1e-10) break; if(j+1>=m) break;
        of.push_back(b); qprev=q; beta=b; for(int i=0;i<n;i++) q[i]=w[i]/b;
    }
    auto cnt=[&](double x)->int{ int c=0; double d=al[0]-x; if(d<0)c++;
        for(int i=1;i<k;i++){ if(fabs(d)<1e-300)d=(d<0?-1e-300:1e-300);
            d=(al[i]-x)-of[i-1]*of[i-1]/d; if(d<0)c++; } return c; };
    double glo=1e300,ghi=-1e300;
    for(int i=0;i<k;i++){ double r=(i>0?fabs(of[i-1]):0)+(i<k-1?fabs(of[i]):0);
        glo=min(glo,al[i]-r); ghi=max(ghi,al[i]+r); }
    double lo=glo-1,hi=ghi+1;
    for(int it=0;it<200&&hi-lo>1e-13*max(1.0,fabs(hi));it++){ double md=0.5*(lo+hi);
        if(cnt(md)>=1) hi=md; else lo=md; }
    vector<double> s=tri_eigvec_small(al,of,hi);
    for(int i=0;i<n;i++){ double v=0; for(int j=0;j<k;j++) v+=Q[j][i]*s[j]; f[i]=v; }
    double mean=0; for(double v:f) mean+=v; mean/=n; for(auto&v:f) v-=mean;
    return f;
}

int main(){
    int n,m; long long B;
    if(scanf("%d %d %lld",&n,&m,&B)!=3) return 0;
    N=n;
    vector<vector<int>> adj(n);        // ORIGINAL graph (for costs)
    vector<vector<int>> cur(n);        // current augmented graph
    auto key=[&](int a,int b)->long long{ if(a>b) swap(a,b); return (long long)a*(n+1)+b; };
    unordered_set<long long> eset; eset.reserve(2*m+16);
    vector<pair<int,int>> curedges;
    for(int i=0;i<m;i++){ int u,v; scanf("%d %d",&u,&v); u--; v--;
        adj[u].push_back(v); adj[v].push_back(u);
        cur[u].push_back(v); cur[v].push_back(u);
        eset.insert(key(u,v)); curedges.push_back({u,v}); }

    unordered_map<int,vector<int>> distc;
    auto bfs=[&](int s)->vector<int>&{ auto it=distc.find(s); if(it!=distc.end())return it->second;
        vector<int> d(n,-1); d[s]=0; queue<int>q; q.push(s);
        while(!q.empty()){int x=q.front();q.pop(); for(int y:adj[x]) if(d[y]<0){d[y]=d[x]+1;q.push(y);}}
        return distc.emplace(s,move(d)).first->second; };

    unordered_set<long long> nset;
    vector<pair<int,int>> out; long long spent=0;
    int msteps = (n<=200?min(n-1,120):100);
    int recompute_every = 10, maxrounds = 48;

    for(int round=0; round<maxrounds && spent<B; round++){
        vector<double> f=fiedler(cur,msteps);
        vector<double> sf=f; nth_element(sf.begin(),sf.begin()+n/2,sf.end());
        double med=sf[n/2];
        vector<char> side(n); for(int i=0;i<n;i++) side[i]=(f[i]>=med)?1:0;
        // candidate gateway-crossing pairs
        unordered_set<long long> seen; seen.reserve(1<<12);
        vector<pair<int,int>> cand;
        for(auto&pr:curedges){ int a=pr.first,b=pr.second; if(side[a]==side[b]) continue;
            // sides: keep a on side[a], b on side[b]
            vector<int> La; La.push_back(a); for(int x:cur[a]) if(side[x]==side[a]) La.push_back(x);
            vector<int> Rb; Rb.push_back(b); for(int y:cur[b]) if(side[y]==side[b]) Rb.push_back(y);
            for(int x:La) for(int y:Rb){ if(x==y) continue; long long kk=key(x,y);
                if(eset.count(kk)||nset.count(kk)||seen.count(kk)) continue; seen.insert(kk);
                cand.push_back({x,y}); }
        }
        if(cand.empty()) break;
        // score by (df)^2 / cost, best first
        vector<tuple<double,int,int,int>> sc; sc.reserve(cand.size());
        for(auto&pr:cand){ int u=pr.first,v=pr.second; int s0=(adj[u].size()<=adj[v].size()?u:v);
            int o=(s0==u?v:u); int d=bfs(s0)[o]; if(d<=0) continue;
            double df=f[u]-f[v]; sc.push_back({df*df/d,u,v,d}); }
        sort(sc.begin(),sc.end(),[](auto&A,auto&B){return get<0>(A)>get<0>(B);});
        int added=0;
        for(auto&t:sc){ if(spent>=B||added>=recompute_every) break;
            int u=get<1>(t),v=get<2>(t),d=get<3>(t); long long kk=key(u,v);
            if(nset.count(kk)) continue;
            if(spent+d<=B){ out.push_back({u,v}); nset.insert(kk); spent+=d; added++;
                cur[u].push_back(v); cur[v].push_back(u); curedges.push_back({u,v}); } }
        if(added==0) break;
    }

    printf("%d\n",(int)out.size());
    for(auto&pr:out) printf("%d %d\n",pr.first+1,pr.second+1);
    return 0;
}
