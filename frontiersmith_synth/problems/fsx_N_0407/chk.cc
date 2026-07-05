// Checker / scorer for "Nimbus Swarm: Submodular Airspace Coverage".
// Composite objective: drone->zone matching (capacities) + geometric energy
// knapsack + SATURATING (submodular) feature coverage.
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll isqrtll(ll v){
    if(v<=0) return 0;
    ll x=(ll)sqrtl((long double)v);
    while(x>0 && x*x>v) x--;
    while((x+1)*(x+1)<=v) x++;
    return x;
}

int D,Z,F;
ll E;
vector<ll> zx,zy; vector<int> zcap;
vector<unordered_map<int,pair<int,int>>> dem; // zone -> feature -> (w,s)
vector<ll> dx,dy;
vector<vector<pair<int,int>>> pay;            // drone -> list of (feature, amount)

static inline ll energy(int d,int z){
    ll ex=dx[d]-zx[z], ey=dy[d]-zy[z];
    return 1 + isqrtll(ex*ex + ey*ey);
}

// value contributed by a SINGLE drone d placed alone on zone z (saturated)
static ll singleValue(int d,int z){
    ll val=0;
    auto &mp=dem[z];
    for(auto &ga: pay[d]){
        auto it=mp.find(ga.first);
        if(it!=mp.end()){
            int w=it->second.first, s=it->second.second;
            val += (ll)w * min((ll)s, (ll)ga.second);
        }
    }
    return val;
}

int main(int argc,char**argv){
    registerTestlibCmd(argc,argv);

    D=inf.readInt(); Z=inf.readInt(); F=inf.readInt(); E=inf.readLong();
    zx.assign(Z+1,0); zy.assign(Z+1,0); zcap.assign(Z+1,0);
    dem.assign(Z+1,{});
    for(int z=1;z<=Z;z++){
        zx[z]=inf.readLong(); zy[z]=inf.readLong(); zcap[z]=inf.readInt();
        int L=inf.readInt();
        for(int j=0;j<L;j++){
            int f=inf.readInt(), w=inf.readInt(), s=inf.readInt();
            dem[z][f]=make_pair(w,s);
        }
    }
    dx.assign(D+1,0); dy.assign(D+1,0); pay.assign(D+1,{});
    for(int d=1;d<=D;d++){
        dx[d]=inf.readLong(); dy[d]=inf.readLong();
        int K=inf.readInt();
        pay[d].reserve(K);
        for(int j=0;j<K;j++){
            int g=inf.readInt(), a=inf.readInt();
            pay[d].push_back(make_pair(g,a));
        }
    }

    // ---- internal baseline B = best feasible SINGLE assignment (one drone, one zone) ----
    ll B=0;
    for(int d=1;d<=D;d++){
        for(int z=1;z<=Z;z++){
            if(zcap[z]<1) continue;
            if(energy(d,z)>E) continue;
            ll v=singleValue(d,z);
            if(v>B) B=v;
        }
    }
    if(B<=0) quitf(_fail,"bad instance: no positive feasible single assignment (B=%lld)",B);

    // ---- read participant assignment ----
    int A=ouf.readInt(0,D,"A");
    vector<char> usedDrone(D+1,0);
    vector<int> cnt(Z+1,0);
    vector<int> touched;
    // per-zone coverage accumulation
    vector<unordered_map<int,ll>> cov(Z+1);
    ll totEnergy=0;
    for(int i=0;i<A;i++){
        int d=ouf.readInt(1,D,"drone");
        int z=ouf.readInt(1,Z,"zone");
        if(usedDrone[d]) quitf(_wa,"drone %d assigned more than once",d);
        usedDrone[d]=1;
        if(cnt[z]==0) touched.push_back(z);
        cnt[z]++;
        if(cnt[z]>zcap[z]) quitf(_wa,"zone %d exceeds capacity %d",z,zcap[z]);
        totEnergy += energy(d,z);
        if(totEnergy>E) quitf(_wa,"total energy %lld exceeds budget %lld",totEnergy,E);
        // accumulate coverage for demanded features only
        auto &mp=dem[z];
        auto &cz=cov[z];
        for(auto &ga: pay[d]){
            if(mp.find(ga.first)!=mp.end()) cz[ga.first]+=ga.second;
        }
    }
    if(!ouf.seekEof()) quitf(_wa,"trailing output tokens");

    // ---- objective F = sum over zones,features of w * min(s, coverage) ----
    ll Fval=0;
    for(int z: touched){
        auto &mp=dem[z];
        for(auto &kv: cov[z]){
            auto it=mp.find(kv.first);
            int w=it->second.first, s=it->second.second;
            Fval += (ll)w * min((ll)s, kv.second);
        }
    }

    double sc = min(1000.0, 100.0 * (double)Fval / (double)max((ll)1,B));
    quitp(sc/1000.0, "OK F=%lld B=%lld Ratio: %.6f", Fval, B, sc/1000.0);
    return 0;
}
