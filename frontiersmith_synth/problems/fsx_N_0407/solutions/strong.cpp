// TIER: strong
// gain/energy-ratio greedy (knapsack-aware) followed by seeded randomized local
// search (add / swap-drone / move / remove) that escapes the overlap trap by
// spreading complementary drones and reclaiming budget from saturated picks.
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

static inline ll energyOf(int d,int z){ ll ex=dx[d]-zx[z],ey=dy[d]-zy[z]; return 1+isqrtll(ex*ex+ey*ey); }

// value of zone z given the multiset of assigned drones on it
static ll zoneValueOf(const vector<int>&drs){
    // drs are assigned to a single zone; caller supplies zone id via closure alt -> we recompute inline
    return 0; // placeholder (unused)
}

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

    vector<vector<int>> zdr(Z+1);            // assigned drones per zone
    vector<int> assignZone(D+1,0);           // 0 = unassigned
    ll rem=E;

    auto zoneVal=[&](int z)->ll{
        auto &mp=dem[z]; unordered_map<int,ll> cov;
        for(int d: zdr[z]) for(auto &ga:pay[d]) if(mp.count(ga.first)) cov[ga.first]+=ga.second;
        ll v=0; for(auto &kv:cov){ auto it=mp.find(kv.first); v+=(ll)it->second.first*min((ll)it->second.second,kv.second); }
        return v;
    };
    auto zoneValWith=[&](int z,int extra)->ll{ // value if `extra` also assigned to z
        auto &mp=dem[z]; unordered_map<int,ll> cov;
        for(int d: zdr[z]) for(auto &ga:pay[d]) if(mp.count(ga.first)) cov[ga.first]+=ga.second;
        if(extra) for(auto &ga:pay[extra]) if(mp.count(ga.first)) cov[ga.first]+=ga.second;
        ll v=0; for(auto &kv:cov){ auto it=mp.find(kv.first); v+=(ll)it->second.first*min((ll)it->second.second,kv.second); }
        return v;
    };

    // ---- ratio greedy ----
    {
        vector<char> used(D+1,0); vector<int> cnt(Z+1,0);
        while(true){
            double bestR=-1; ll bestGain=0; int bd=-1,bz=-1;
            for(int d=1;d<=D;d++){ if(used[d]) continue;
                for(int z=1;z<=Z;z++){ if(cnt[z]>=zcap[z]) continue;
                    ll e=energyOf(d,z); if(e>rem) continue;
                    ll g=zoneValWith(z,d)-zoneVal(z);
                    if(g<=0) continue;
                    double r=(double)g/(double)e;
                    if(r>bestR){ bestR=r; bestGain=g; bd=d; bz=z; }
                }
            }
            if(bd<0) break;
            used[bd]=1; cnt[bz]++; rem-=energyOf(bd,bz);
            zdr[bz].push_back(bd); assignZone[bd]=bz; (void)bestGain;
        }
    }

    // current total value
    auto totalVal=[&]()->ll{ ll v=0; for(int z=1;z<=Z;z++) if(!zdr[z].empty()) v+=zoneVal(z); return v; };
    ll curF=totalVal();

    // ---- seeded randomized local search ----
    mt19937 rng(987654321u);
    auto zcnt=[&](int z){ return (int)zdr[z].size(); };
    int iters = min(60000, 2000 + D*20);
    for(int it=0; it<iters; ++it){
        int mv=rng()%4;
        if(mv==0){ // ADD an unassigned drone to a zone
            int d=rng()%D+1; if(assignZone[d]) continue;
            int z=rng()%Z+1; if(zcnt(z)>=zcap[z]) continue;
            ll e=energyOf(d,z); if(e>rem) continue;
            ll delta=zoneValWith(z,d)-zoneVal(z);
            if(delta>0){ zdr[z].push_back(d); assignZone[d]=z; rem-=e; curF+=delta; }
        } else if(mv==1){ // SWAP an assigned drone for an unassigned one on same zone
            int d=rng()%D+1; int z=assignZone[d]; if(!z) continue;
            int d2=rng()%D+1; if(assignZone[d2]) continue;
            ll eOld=energyOf(d,z), eNew=energyOf(d2,z);
            if(eNew-eOld>rem) continue;
            // remove d
            auto &v=zdr[z]; v.erase(find(v.begin(),v.end(),d));
            ll withoutD=zoneVal(z);
            v.push_back(d2);
            ll withD2=zoneVal(z);
            // compare against original (d present)
            v.pop_back(); v.push_back(d);
            ll withD=zoneVal(z);
            if(withD2>withD || (withD2==withD && eNew<eOld)){
                v.pop_back(); v.push_back(d2);
                assignZone[d]=0; assignZone[d2]=z; rem-=(eNew-eOld);
                curF += (withD2-withD);
            }
            (void)withoutD;
        } else if(mv==2){ // MOVE assigned drone to a different zone
            int d=rng()%D+1; int z=assignZone[d]; if(!z) continue;
            int z2=rng()%Z+1; if(z2==z||zcnt(z2)>=zcap[z2]) continue;
            ll eOld=energyOf(d,z), eNew=energyOf(d,z2);
            if(eNew-eOld>rem) continue;
            ll oldZ=zoneVal(z);
            auto &v=zdr[z]; v.erase(find(v.begin(),v.end(),d));
            ll newZ=zoneVal(z);
            ll gain2=zoneValWith(z2,d)-zoneVal(z2);
            ll delta=(newZ-oldZ)+gain2;
            if(delta>0){ zdr[z2].push_back(d); assignZone[d]=z2; rem-=(eNew-eOld); curF+=delta; }
            else { v.push_back(d); } // revert
        } else { // REMOVE a near-useless assigned drone (reclaim budget)
            int d=rng()%D+1; int z=assignZone[d]; if(!z) continue;
            ll oldZ=zoneVal(z);
            auto &v=zdr[z]; v.erase(find(v.begin(),v.end(),d));
            ll newZ=zoneVal(z);
            ll loss=oldZ-newZ;
            if(loss==0){ assignZone[d]=0; rem+=energyOf(d,z); }
            else v.push_back(d); // keep
        }
    }

    vector<pair<int,int>> out;
    for(int d=1;d<=D;d++) if(assignZone[d]) out.push_back({d,assignZone[d]});
    printf("%d\n",(int)out.size());
    for(auto &p:out) printf("%d %d\n",p.first,p.second);
    (void)zoneValueOf;
    return 0;
}
