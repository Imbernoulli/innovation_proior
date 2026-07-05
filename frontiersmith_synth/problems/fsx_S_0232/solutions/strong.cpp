// TIER: strong
// Single-tour prize-collecting insertion + or-opt local search over a
// load-weighted pickup-delivery metric. Batching into ONE tour removes the
// per-order depot round-trips that the greedy pays, and local moves
// (relocate / drop / add) further cut load-weighted travel. Guarded to be
// no worse than the greedy baseline.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static inline ll manh(ll x1,ll y1,ll x2,ll y2){ return llabs(x1-x2)+llabs(y1-y2); }

int P,Q; ll DX,DY;
vector<ll> ax,ay,bx,by,q,w;

struct Ev{ int t,i; };

// simulate a sequence; returns {feasible, load-weighted travel}
pair<bool,ll> simTravel(const vector<Ev>&s){
    ll cur=0, px=DX, py=DY, D=0;
    for(const Ev&e:s){
        ll tx,ty;
        if(e.t==0){ tx=ax[e.i]; ty=ay[e.i]; } else { tx=bx[e.i]; ty=by[e.i]; }
        D += manh(px,py,tx,ty)*(1+cur);
        if(e.t==0){ cur+=q[e.i]; if(cur>Q) return {false,0}; }
        else       { cur-=q[e.i]; if(cur<0) return {false,0}; }
        px=tx; py=ty;
    }
    if(cur!=0) return {false,0};
    D += manh(px,py,DX,DY)*(1+cur);
    return {true,D};
}

// F = travel + penalties for unserved. Infeasible -> huge.
ll computeF(const vector<Ev>&s){
    auto pr=simTravel(s);
    if(!pr.first) return LLONG_MAX/4;
    vector<char> served(P,0);
    for(const Ev&e:s) if(e.t==0) served[e.i]=1;
    ll F=pr.second;
    for(int i=0;i<P;i++) if(!served[i]) F+=w[i];
    return F;
}

int main(){
    if(scanf("%d %d",&P,&Q)!=2) return 0;
    scanf("%lld %lld",&DX,&DY);
    ax.resize(P);ay.resize(P);bx.resize(P);by.resize(P);q.resize(P);w.resize(P);
    for(int i=0;i<P;i++) scanf("%lld %lld %lld %lld %lld %lld",&ax[i],&ay[i],&bx[i],&by[i],&q[i],&w[i]);

    // ---------- greedy baseline route (isolated round-trips) ----------
    vector<pair<ll,int>> ben;
    for(int i=0;i<P;i++){
        ll solo = manh(DX,DY,ax[i],ay[i])
                + manh(ax[i],ay[i],bx[i],by[i])*(1+q[i])
                + manh(bx[i],by[i],DX,DY);
        if(w[i]-solo>0) ben.push_back({-(w[i]-solo),i});
    }
    sort(ben.begin(),ben.end());
    vector<Ev> gseq;
    for(auto&pr:ben){ gseq.push_back({0,pr.second}); gseq.push_back({1,pr.second}); }

    // ---------- single-tour prize-collecting block insertion ----------
    vector<Ev> seq;
    vector<char> served(P,0);
    while(true){
        ll curF=computeF(seq);
        ll bestNet=0; int bestReq=-1; int bestPos=-1;
        for(int r=0;r<P;r++){
            if(served[r]) continue;
            int n=(int)seq.size();
            for(int pos=0;pos<=n;pos++){
                vector<Ev> cand; cand.reserve(n+2);
                for(int k=0;k<pos;k++) cand.push_back(seq[k]);
                cand.push_back({0,r}); cand.push_back({1,r});
                for(int k=pos;k<n;k++) cand.push_back(seq[k]);
                ll f=computeF(cand);
                if(f>=LLONG_MAX/8) continue;
                ll net=f-curF;
                if(net<bestNet){ bestNet=net; bestReq=r; bestPos=pos; }
            }
        }
        if(bestReq<0) break;
        vector<Ev> ns; int n=(int)seq.size();
        for(int k=0;k<bestPos;k++) ns.push_back(seq[k]);
        ns.push_back({0,bestReq}); ns.push_back({1,bestReq});
        for(int k=bestPos;k<n;k++) ns.push_back(seq[k]);
        seq.swap(ns); served[bestReq]=1;
    }

    // ---------- or-opt local search: drop / relocate / add ----------
    bool improved=true;
    int guard=0;
    while(improved && guard++<10000){
        improved=false;
        ll curF=computeF(seq);

        // 1) drop a served order (remove both its events)
        for(int r=0;r<P && !improved;r++){
            if(!served[r]) continue;
            vector<Ev> base;
            for(const Ev&e:seq) if(e.i!=r) base.push_back(e);
            ll f=computeF(base);
            if(f<curF){ seq.swap(base); served[r]=0; improved=true; }
        }
        if(improved) continue;

        // 2) relocate a served order's block to a better position
        for(int r=0;r<P && !improved;r++){
            if(!served[r]) continue;
            vector<Ev> base;
            for(const Ev&e:seq) if(e.i!=r) base.push_back(e);
            int n=(int)base.size();
            ll bestF=curF; vector<Ev> bestSeq; bool found=false;
            for(int pos=0;pos<=n;pos++){
                vector<Ev> cand; cand.reserve(n+2);
                for(int k=0;k<pos;k++) cand.push_back(base[k]);
                cand.push_back({0,r}); cand.push_back({1,r});
                for(int k=pos;k<n;k++) cand.push_back(base[k]);
                ll f=computeF(cand);
                if(f<bestF){ bestF=f; bestSeq=cand; found=true; }
            }
            if(found){ seq.swap(bestSeq); improved=true; }
        }
        if(improved) continue;

        // 3) add an unserved order at its best block position
        for(int r=0;r<P && !improved;r++){
            if(served[r]) continue;
            int n=(int)seq.size();
            ll bestF=curF; vector<Ev> bestSeq; bool found=false;
            for(int pos=0;pos<=n;pos++){
                vector<Ev> cand; cand.reserve(n+2);
                for(int k=0;k<pos;k++) cand.push_back(seq[k]);
                cand.push_back({0,r}); cand.push_back({1,r});
                for(int k=pos;k<n;k++) cand.push_back(seq[k]);
                ll f=computeF(cand);
                if(f<bestF){ bestF=f; bestSeq=cand; found=true; }
            }
            if(found){ seq.swap(bestSeq); served[r]=1; improved=true; }
        }
    }

    // ---------- pick the better of insertion-LS vs greedy baseline ----------
    ll fStrong=computeF(seq);
    ll fGreedy=computeF(gseq);
    const vector<Ev>& out = (fGreedy<fStrong)? gseq : seq;

    printf("%d\n",(int)out.size());
    for(const Ev&e:out) printf("%d %d\n",e.t,e.i+1);
    return 0;
}
