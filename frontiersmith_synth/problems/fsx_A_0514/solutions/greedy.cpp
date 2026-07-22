// TIER: greedy
// Textbook Fiedler greedy: compute the Fiedler vector ONCE, then repeatedly bridge the two
// most-separated platforms (largest |f_u - f_v|) across the top Cheeger cut, spending the whole
// budget there. This is the obvious first attempt -- but those extreme platforms sit DEEP in
// their clusters, hop-far from the gateways, so each bridge is expensive; and it never recomputes,
// so it pours everything into the single visible cut and is blind to the bottleneck migrating to
// deeper scales once the top cut is repaired.
#include <bits/stdc++.h>
using namespace std;

int N;
static vector<double> tri_eigvec_small(const vector<double>&al,const vector<double>&of,double mu){
    int k=al.size(); vector<double> y(k,1.0);
    for(int iter=0;iter<3;iter++){
        // solve (T - mu I) x = y  (Thomas), guard tiny pivots -> amplifies target eigenvector
        vector<double> c(k,0.0), d(k,0.0);
        double piv=al[0]-mu; if(fabs(piv)<1e-30) piv=(piv<0?-1e-30:1e-30);
        c[0]=(k>1?of[0]:0.0)/piv; d[0]=y[0]/piv;
        for(int i=1;i<k;i++){
            double den=(al[i]-mu)-of[i-1]*c[i-1];
            if(fabs(den)<1e-30) den=(den<0?-1e-30:1e-30);
            c[i]=(i<k-1?of[i]:0.0)/den;
            d[i]=(y[i]-of[i-1]*d[i-1])/den;
        }
        vector<double> x(k); x[k-1]=d[k-1];
        for(int i=k-2;i>=0;i--) x[i]=d[i]-c[i]*x[i+1];
        double nn=0; for(double v:x) nn+=v*v; nn=sqrt(nn); if(nn<1e-300) nn=1;
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
    // smallest eigenvalue via bisection
    auto cnt=[&](double x)->int{ int c=0; double d=al[0]-x; if(d<0)c++;
        for(int i=1;i<k;i++){ if(fabs(d)<1e-300)d=(d<0?-1e-300:1e-300);
            d=(al[i]-x)-of[i-1]*of[i-1]/d; if(d<0)c++; } return c; };
    double glo=1e300,ghi=-1e300;
    for(int i=0;i<k;i++){ double r=(i>0?fabs(of[i-1]):0)+(i<k-1?fabs(of[i]):0);
        glo=min(glo,al[i]-r); ghi=max(ghi,al[i]+r); }
    double lo=glo-1,hi=ghi+1;
    for(int it=0;it<200&&hi-lo>1e-13*max(1.0,fabs(hi));it++){ double md=0.5*(lo+hi);
        if(cnt(md)>=1) hi=md; else lo=md; }
    double mu=hi;
    vector<double> s=tri_eigvec_small(al,of,mu);
    for(int i=0;i<n;i++){ double v=0; for(int j=0;j<k;j++) v+=Q[j][i]*s[j]; f[i]=v; }
    double mean=0; for(double v:f) mean+=v; mean/=n; for(auto&v:f) v-=mean;
    return f;
}

int main(){
    int n,m; long long B;
    if(scanf("%d %d %lld",&n,&m,&B)!=3) return 0;
    N=n;
    vector<vector<int>> adj(n);
    vector<vector<char>> dummy;
    auto key=[&](int a,int b)->long long{ if(a>b) swap(a,b); return (long long)a*(n+1)+b; };
    unordered_set<long long> eset; eset.reserve(2*m+16);
    for(int i=0;i<m;i++){ int u,v; scanf("%d %d",&u,&v); u--; v--;
        adj[u].push_back(v); adj[v].push_back(u); eset.insert(key(u,v)); }

    vector<double> f=fiedler(adj,150);
    vector<int> idx(n); iota(idx.begin(),idx.end(),0);
    sort(idx.begin(),idx.end(),[&](int a,int b){ return f[a]<f[b]; });
    // lo = most negative half, hi = most positive half (descending extremeness)
    int half=n/2;
    vector<int> lo(idx.begin(),idx.begin()+half);
    vector<int> hi(idx.begin()+half,idx.end()); reverse(hi.begin(),hi.end());

    unordered_map<int,vector<int>> distc;
    auto bfs=[&](int s)->vector<int>&{ auto it=distc.find(s); if(it!=distc.end())return it->second;
        vector<int> d(n,-1); d[s]=0; queue<int>q; q.push(s);
        while(!q.empty()){int x=q.front();q.pop(); for(int y:adj[x]) if(d[y]<0){d[y]=d[x]+1;q.push(y);}}
        return distc.emplace(s,move(d)).first->second; };

    unordered_set<long long> nset;
    vector<pair<int,int>> out; long long spent=0;
    int li=0, hj=0;
    while(spent<B && li<(int)lo.size() && !hi.empty()){
        int u=lo[li], v=hi[hj%hi.size()];
        long long kk=key(u,v);
        if(u!=v && !eset.count(kk) && !nset.count(kk)){
            int c=bfs(u)[v];
            if(c>0 && spent+c<=B){ out.push_back({u,v}); nset.insert(kk); spent+=c; }
        }
        li++; hj++;
    }
    printf("%d\n",(int)out.size());
    for(auto&pr:out) printf("%d %d\n",pr.first+1,pr.second+1);
    return 0;
}
