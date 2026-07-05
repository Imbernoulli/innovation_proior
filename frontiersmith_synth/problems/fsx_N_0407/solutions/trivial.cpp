// TIER: trivial
// Assign the single best feasible (drone,zone) pair -> F == checker baseline B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll isqrtll(ll v){ if(v<=0)return 0; ll x=(ll)sqrtl((long double)v);
    while(x>0&&x*x>v)x--; while((x+1)*(x+1)<=v)x++; return x; }

int main(){
    int D,Z,F; ll E;
    scanf("%d %d %d %lld",&D,&Z,&F,&E);
    vector<ll> zx(Z+1),zy(Z+1); vector<int> zcap(Z+1);
    vector<unordered_map<int,pair<int,int>>> dem(Z+1);
    for(int z=1;z<=Z;z++){
        ll a,b; int c,L; scanf("%lld %lld %d %d",&a,&b,&c,&L);
        zx[z]=a; zy[z]=b; zcap[z]=c;
        for(int j=0;j<L;j++){ int f,w,s; scanf("%d %d %d",&f,&w,&s); dem[z][f]={w,s}; }
    }
    vector<ll> dx(D+1),dy(D+1);
    vector<vector<pair<int,int>>> pay(D+1);
    for(int d=1;d<=D;d++){
        ll a,b; int K; scanf("%lld %lld %d",&a,&b,&K); dx[d]=a; dy[d]=b;
        for(int j=0;j<K;j++){ int g,am; scanf("%d %d",&g,&am); pay[d].push_back({g,am}); }
    }
    auto energy=[&](int d,int z){ ll ex=dx[d]-zx[z],ey=dy[d]-zy[z]; return 1+isqrtll(ex*ex+ey*ey); };
    ll best=-1; int bd=-1,bz=-1;
    for(int d=1;d<=D;d++) for(int z=1;z<=Z;z++){
        if(zcap[z]<1) continue; if(energy(d,z)>E) continue;
        ll v=0; auto &mp=dem[z];
        for(auto &ga:pay[d]){ auto it=mp.find(ga.first); if(it!=mp.end()) v+=(ll)it->second.first*min((ll)it->second.second,(ll)ga.second); }
        if(v>best){ best=v; bd=d; bz=z; }
    }
    if(bd<0){ printf("0\n"); return 0; }
    printf("1\n%d %d\n",bd,bz);
    return 0;
}
