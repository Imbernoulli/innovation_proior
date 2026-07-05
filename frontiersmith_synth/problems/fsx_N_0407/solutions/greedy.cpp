// TIER: greedy
// Greedy-by-absolute-marginal: repeatedly add the (drone,zone) pair with the
// largest marginal coverage gain that still fits budget+capacity. Falls into the
// overlap trap (chases fat popular demand, saturates, wastes energy).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll isqrtll(ll v){ if(v<=0)return 0; ll x=(ll)sqrtl((long double)v);
    while(x>0&&x*x>v)x--; while((x+1)*(x+1)<=v)x++; return x; }

int D,Z,F; ll E;
vector<ll> zx,zy; vector<int> zcap;
vector<unordered_map<int,pair<int,int>>> dem;
vector<ll> dx,dy;
vector<vector<pair<int,int>>> pay;

int main(){
    scanf("%d %d %d %lld",&D,&Z,&F,&E);
    zx.assign(Z+1,0); zy.assign(Z+1,0); zcap.assign(Z+1,0); dem.assign(Z+1,{});
    for(int z=1;z<=Z;z++){
        ll a,b; int c,L; scanf("%lld %lld %d %d",&a,&b,&c,&L);
        zx[z]=a; zy[z]=b; zcap[z]=c;
        for(int j=0;j<L;j++){ int f,w,s; scanf("%d %d %d",&f,&w,&s); dem[z][f]={w,s}; }
    }
    dx.assign(D+1,0); dy.assign(D+1,0); pay.assign(D+1,{});
    for(int d=1;d<=D;d++){
        ll a,b; int K; scanf("%lld %lld %d",&a,&b,&K); dx[d]=a; dy[d]=b;
        for(int j=0;j<K;j++){ int g,am; scanf("%d %d",&g,&am); pay[d].push_back({g,am}); }
    }
    auto energy=[&](int d,int z){ ll ex=dx[d]-zx[z],ey=dy[d]-zy[z]; return 1+isqrtll(ex*ex+ey*ey); };

    vector<char> used(D+1,0);
    vector<int> cnt(Z+1,0);
    vector<unordered_map<int,ll>> cov(Z+1);
    ll rem=E;
    vector<pair<int,int>> chosen;

    // marginal gain of adding drone d to zone z given current coverage
    auto gain=[&](int d,int z)->ll{
        auto &mp=dem[z]; auto &cz=cov[z]; ll g=0;
        for(auto &ga:pay[d]){ auto it=mp.find(ga.first); if(it==mp.end()) continue;
            int w=it->second.first,s=it->second.second;
            ll cur= (cz.count(ga.first)? cz[ga.first]:0);
            ll add= min((ll)s,cur+ga.second)-min((ll)s,cur);
            g += (ll)w*add; }
        return g;
    };

    while(true){
        ll bestG=0; int bd=-1,bz=-1;
        for(int d=1;d<=D;d++){ if(used[d]) continue;
            for(int z=1;z<=Z;z++){ if(cnt[z]>=zcap[z]) continue;
                ll e=energy(d,z); if(e>rem) continue;
                ll g=gain(d,z);
                if(g>bestG){ bestG=g; bd=d; bz=z; }
            }
        }
        if(bd<0) break;
        used[bd]=1; cnt[bz]++; rem-=energy(bd,bz);
        auto &mp=dem[bz]; auto &cz=cov[bz];
        for(auto &ga:pay[bd]) if(mp.count(ga.first)) cz[ga.first]+=ga.second;
        chosen.push_back({bd,bz});
    }

    printf("%d\n",(int)chosen.size());
    for(auto &p:chosen) printf("%d %d\n",p.first,p.second);
    return 0;
}
