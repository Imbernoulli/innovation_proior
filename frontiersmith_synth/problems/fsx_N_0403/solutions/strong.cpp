// TIER: strong
// Density knapsack + residue repair. Rigs are packed into loops by weight/coolant density
// (more thermal weight per unit coolant), the coolant budget is topped off with cheap rigs to
// lift the fill bonus, and the quadratic-residue lock is repaired by the least-costly
// ADD-a-rig (net weight gain) before falling back to dropping a rig. This salvages the heavy
// rigs that are unstable alone but stable when grouped.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll powmod(ll a, ll e, ll m){ a%=m; if(a<0)a+=m; ll r=1%m; while(e){ if(e&1) r=(__int128)r*a%m; a=(__int128)a*a%m; e>>=1;} return r; }
static bool isNZQR(ll s, ll p){ s%=p; if(s<0)s+=p; if(s==0) return false; return powmod(s,(p-1)/2,p)==1; }

int main(){
    ll N,M,C,p;
    if(scanf("%lld %lld %lld %lld",&N,&M,&C,&p)!=4) return 0;
    vector<ll> w(N+1),c(N+1),k(N+1);
    for(ll i=1;i<=N;i++) scanf("%lld %lld %lld",&w[i],&c[i],&k[i]);

    vector<char> used(N+1,0);

    // density order (knapsack heuristic)
    vector<ll> dens(N);
    for(ll i=0;i<N;i++) dens[i]=i+1;
    sort(dens.begin(),dens.end(),[&](ll a,ll b){
        // compare w[a]/c[a] vs w[b]/c[b] without floating error
        __int128 l=(__int128)w[a]*c[b], r=(__int128)w[b]*c[a];
        if(l!=r) return l>r;
        return w[a]>w[b];
    });
    // cheap-coolant order (fillers / adjusters)
    vector<ll> cheap(N);
    for(ll i=0;i<N;i++) cheap[i]=i+1;
    sort(cheap.begin(),cheap.end(),[&](ll a,ll b){ if(c[a]!=c[b]) return c[a]<c[b]; return w[a]>w[b]; });

    vector<vector<ll>> loops;
    ll dp=0;

    auto stable_add_scan=[&](ll ksum,ll U)->ll{
        // find an unused cheap rig that fits budget and flips residue to stable (net weight gain)
        int attempts=0;
        for(ll t=0;t<(ll)cheap.size() && attempts<80;t++){
            ll id=cheap[t];
            if(used[id]) continue;
            attempts++;
            if(U + c[id] > C) continue;
            if(isNZQR((ksum + k[id])%p, p)) return id;
        }
        return -1;
    };

    while((ll)loops.size()<M && dp<N){
        vector<ll> cur; ll U=0, ksum=0;
        // primary density fill
        while(dp<N){
            ll id=dens[dp];
            if(used[id]){ dp++; continue; }
            if(U + c[id] > C){ if(!cur.empty()) break; dp++; continue; }
            cur.push_back(id); used[id]=1; U+=c[id]; ksum=(ksum+k[id])%p; dp++;
            if((ll)cur.size()>=8) break;
        }
        if(cur.empty()) continue;

        // top-off the coolant budget with cheap unused rigs (raises the fill bonus)
        {
            int added=0, scan=0;
            for(ll t=0;t<(ll)cheap.size() && added<3 && scan<200;t++){
                ll id=cheap[t];
                if(used[id]) continue;
                scan++;
                if(U + c[id] > C) continue;
                cur.push_back(id); used[id]=1; U+=c[id]; ksum=(ksum+k[id])%p; added++;
            }
        }

        // repair residue lock
        if(!isNZQR(ksum,p)){
            ll addId=stable_add_scan(ksum,U);
            if(addId!=-1){
                cur.push_back(addId); used[addId]=1; U+=c[addId]; ksum=(ksum+k[addId])%p;
            } else {
                // drop the min-weight member whose removal makes the loop stable
                ll bestDrop=-1, bestW=LLONG_MAX;
                for(ll id:cur){
                    ll ns=((ksum-k[id])%p+p)%p;
                    if(isNZQR(ns,p) && w[id]<bestW){ bestW=w[id]; bestDrop=id; }
                }
                if(bestDrop!=-1){
                    vector<ll> nc; for(ll id:cur){ if(id==bestDrop){ used[id]=0; } else nc.push_back(id); }
                    cur.swap(nc);
                    ksum=0; U=0; for(ll id:cur){ ksum=(ksum+k[id])%p; U+=c[id]; }
                } else {
                    // fall back: heaviest individually-stable member as a singleton
                    ll pick=-1;
                    for(ll id:cur) if(isNZQR(k[id],p) && (pick==-1||w[id]>w[pick])) pick=id;
                    for(ll id:cur) if(id!=pick) used[id]=0;
                    if(pick==-1) continue;
                    cur.clear(); cur.push_back(pick);
                }
            }
        }
        if(!cur.empty()) loops.push_back(cur);
    }

    printf("%lld\n",(ll)loops.size());
    for(auto& L:loops){
        printf("%lld",(ll)L.size());
        for(ll id:L) printf(" %lld",id);
        printf("\n");
    }
    return 0;
}
