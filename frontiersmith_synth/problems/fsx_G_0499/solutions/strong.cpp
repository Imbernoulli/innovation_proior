// TIER: strong
// Multi-restart, class-aware constructive search with ring saturation and pruning.
// Builds an assembly from several fragment orderings (b-desc, synergy-class-first,
// and many seeded random shuffles), connecting each accepted fragment to the best
// free-valence neighbour, then saturates ring-closing bonds (q + Rb > 0) and prunes
// leaf fragments whose marginal contribution is negative. Keeps the highest-F
// assembly found. Deterministic (fixed RNG seed).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll M,C,Wmax,Lam,Rb;
static vector<ll> w,b,val,p;
static vector<vector<ll>> q;

static inline ll over(ll W){ return max((ll)0, W-Wmax); }

// exact objective for an assembly (mirrors the checker)
static ll scoreAssembly(const vector<ll>& sel, const vector<pair<ll,ll>>& bonds){
    ll frag=0,W=0;
    for(ll i: sel){ frag+=b[i]; W+=w[i]; }
    ll bond=0; for(auto&e:bonds) bond+=q[p[e.first]][p[e.second]];
    ll S=(ll)sel.size(), E=(ll)bonds.size();
    ll rings=E-S+1;
    return frag + bond + Rb*rings - Lam*over(W);
}

struct Build { vector<ll> sel; vector<pair<ll,ll>> bonds; ll F; };

// grow an assembly from a candidate order; connect each accepted fragment to the
// free-valence selected fragment with the best interaction; accept if marginal > 0.
static Build growFrom(const vector<ll>& order){
    vector<char> chosen(M+1,0);
    vector<ll> rem(M+1); for(ll i=1;i<=M;i++) rem[i]=val[i];
    vector<ll> sel; vector<pair<ll,ll>> bonds; set<pair<ll,ll>> bset;
    ll W=0;
    auto addFrag=[&](ll j){ chosen[j]=1; sel.push_back(j); W+=w[j]; };
    auto addBond=[&](ll a,ll c){ bonds.push_back({a,c}); bset.insert({min(a,c),max(a,c)}); rem[a]--; rem[c]--; };

    // seed with first order element
    addFrag(order[0]);
    for(size_t oi=1; oi<order.size(); oi++){
        ll j=order[oi]; if(chosen[j]) continue;
        ll bi=-1,bq=LLONG_MIN;
        for(ll i: sel) if(rem[i]>0){ ll v=q[p[i]][p[j]]; if(v>bq){bq=v;bi=i;} }
        if(bi<0) continue;
        ll delta=b[j]+bq-Lam*(over(W+w[j])-over(W));
        if(delta>0){ addFrag(j); addBond(bi,j); }
    }
    // ring saturation
    ll ns=(ll)sel.size();
    for(ll x=0;x<ns;x++) for(ll y=x+1;y<ns;y++){
        ll a=sel[x],c=sel[y];
        if(rem[a]>0 && rem[c]>0){
            auto key=make_pair(min(a,c),max(a,c));
            if(!bset.count(key) && q[p[a]][p[c]]+Rb>0) addBond(a,c);
        }
    }
    Build B; B.sel=sel; B.bonds=bonds; B.F=scoreAssembly(sel,bonds); return B;
}

int main(){
    if(!(cin>>M>>C>>Wmax>>Lam>>Rb)) return 0;
    w.assign(M+1,0);b.assign(M+1,0);val.assign(M+1,0);p.assign(M+1,0);
    for(ll i=1;i<=M;i++) cin>>w[i]>>b[i]>>val[i]>>p[i];
    q.assign(C, vector<ll>(C));
    for(ll a=0;a<C;a++) for(ll c=0;c<C;c++) cin>>q[a][c];

    ll seed=1,best=LLONG_MIN;
    for(ll i=1;i<=M;i++){ ll v=b[i]-Lam*over(w[i]); if(v>best){best=v;seed=i;} }

    // "synergy potential" of a fragment: best interaction its class can achieve
    vector<ll> synp(M+1,0);
    for(ll i=1;i<=M;i++){ ll mx=LLONG_MIN; for(ll c=0;c<C;c++) mx=max(mx,q[p[i]][c]); synp[i]=mx; }

    vector<Build> cands;

    // order 1: b descending, seed first
    {
        vector<ll> ord; ord.push_back(seed);
        vector<ll> rest; for(ll i=1;i<=M;i++) if(i!=seed) rest.push_back(i);
        sort(rest.begin(),rest.end(),[&](ll x,ll y){ return b[x]>b[y]; });
        for(ll x:rest) ord.push_back(x);
        cands.push_back(growFrom(ord));
    }
    // order 2: synergy-class-first (high synp, then b), seed = best synergy fragment
    {
        vector<ll> ids; for(ll i=1;i<=M;i++) ids.push_back(i);
        sort(ids.begin(),ids.end(),[&](ll x,ll y){
            if(synp[x]!=synp[y]) return synp[x]>synp[y];
            return b[x]>b[y];
        });
        cands.push_back(growFrom(ids));
    }
    // random restarts
    mt19937_64 rng(987654321ULL);
    int RESTARTS = (M<=200)?40:((M<=800)?24:12);
    for(int r=0;r<RESTARTS;r++){
        vector<ll> ord; for(ll i=1;i<=M;i++) ord.push_back(i);
        shuffle(ord.begin(),ord.end(),rng);
        // bias: put a strong seed first with some probability
        if(r%2==0){ for(size_t i=0;i<ord.size();i++) if(ord[i]==seed){ swap(ord[0],ord[i]); break; } }
        cands.push_back(growFrom(ord));
    }

    // pick best
    Build bestB=cands[0];
    for(auto&cb:cands) if(cb.F>bestB.F) bestB=cb;

    // local improvement: prune leaf fragments with negative marginal contribution,
    // then re-saturate rings. Repeat a few passes.
    for(int pass=0; pass<3; pass++){
        // recompute degrees & remaining valence on bestB
        ll S=(ll)bestB.sel.size();
        vector<ll> deg(M+1,0);
        set<pair<ll,ll>> bset;
        for(auto&e:bestB.bonds){ deg[e.first]++; deg[e.second]++; bset.insert({min(e.first,e.second),max(e.first,e.second)}); }
        // try removing a degree-1 (leaf) fragment if it lowers F
        bool changed=false;
        if(S>1){
            for(ll idx=0; idx<S; idx++){
                ll j=bestB.sel[idx];
                if(deg[j]!=1) continue;
                // build candidate without j and its one bond
                vector<ll> nsel; for(ll s:bestB.sel) if(s!=j) nsel.push_back(s);
                vector<pair<ll,ll>> nb; for(auto&e:bestB.bonds) if(e.first!=j&&e.second!=j) nb.push_back(e);
                ll nf=scoreAssembly(nsel,nb);
                if(nf>bestB.F){ bestB.sel=nsel; bestB.bonds=nb; bestB.F=nf; changed=true; break; }
            }
        }
        // re-saturate remaining rings
        {
            ll ns=(ll)bestB.sel.size();
            vector<ll> rem(M+1,0); for(ll s:bestB.sel) rem[s]=val[s];
            set<pair<ll,ll>> bs;
            for(auto&e:bestB.bonds){ rem[e.first]--; rem[e.second]--; bs.insert({min(e.first,e.second),max(e.first,e.second)}); }
            for(ll x=0;x<ns;x++) for(ll y=x+1;y<ns;y++){
                ll a=bestB.sel[x],c=bestB.sel[y];
                if(rem[a]>0&&rem[c]>0){
                    auto key=make_pair(min(a,c),max(a,c));
                    if(!bs.count(key) && q[p[a]][p[c]]+Rb>0){ bestB.bonds.push_back({a,c}); bs.insert(key); rem[a]--; rem[c]--; changed=true; }
                }
            }
            bestB.F=scoreAssembly(bestB.sel,bestB.bonds);
        }
        if(!changed) break;
    }

    // fallback: never worse than best single fragment
    if(bestB.sel.empty() || bestB.F < best){
        printf("1\n%lld\n0\n", seed); return 0;
    }

    printf("%lld\n",(ll)bestB.sel.size());
    for(size_t i=0;i<bestB.sel.size();i++) printf("%lld%c", bestB.sel[i], i+1==bestB.sel.size()?'\n':' ');
    printf("%lld\n",(ll)bestB.bonds.size());
    for(auto&e:bestB.bonds) printf("%lld %lld\n", e.first, e.second);
    return 0;
}
