// TIER: strong
// Insight: because toppling is ABELIAN, the target is a linear SUPERPOSITION of drops
// spread over many sources -- not a single tuned pile. So build the answer as a
// combination: coordinate ascent that repeatedly commits, at whichever source helps
// most, an extra chunk of grains, re-stabilizing the CURRENT (already-stable) pile
// incrementally each time. Combining sources reconstructs the multi-lobe target that
// any single pile is blind to. (Still a heuristic: it hill-climbs, not solves, the
// integer toppling system -- leaving room above it.)
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N;
static inline int IDX(int r,int c){return r*N+c;}

void stabilize(vector<int>&g, const vector<char>&sink){
    int NN=N*N; vector<char> inq(NN,0); deque<int> q;
    for(int i=0;i<NN;i++) if(!sink[i]&&g[i]>=4){q.push_back(i);inq[i]=1;}
    static const int dr[4]={-1,1,0,0}, dc[4]={0,0,-1,1};
    while(!q.empty()){
        int c=q.front();q.pop_front();inq[c]=0;
        if(sink[c]||g[c]<4) continue;
        int t=g[c]/4; g[c]-=4*t; int r=c/N, cc=c%N;
        for(int d=0;d<4;d++){
            int nr=r+dr[d], nc=cc+dc[d];
            if(nr<0||nr>=N||nc<0||nc>=N) continue;
            int ni=nr*N+nc; if(sink[ni]) continue;
            g[ni]+=t; if(g[ni]>=4&&!inq[ni]){inq[ni]=1;q.push_back(ni);}
        }
    }
}

int main(){
    int K; ll B;
    if(scanf("%d %d %lld",&N,&K,&B)!=3) return 0;
    int NN=N*N;
    vector<char> sink(NN,0);
    int nSink; scanf("%d",&nSink);
    for(int i=0;i<nSink;i++){int r,c;scanf("%d %d",&r,&c);sink[IDX(r,c)]=1;}
    vector<int> src(K);
    for(int i=0;i<K;i++){int r,c;scanf("%d %d",&r,&c);src[i]=IDX(r,c);}
    vector<int> T(NN),W(NN);
    for(int i=0;i<NN;i++) scanf("%d",&T[i]);
    for(int i=0;i<NN;i++) scanf("%d",&W[i]);

    auto score=[&](const vector<int>&g)->ll{
        ll F=0; for(int i=0;i<NN;i++) if(!sink[i]&&g[i]==T[i]) F+=W[i]; return F;
    };

    ll u = max((ll)4, (ll)llround(0.10*NN/(double)max(1,K)));
    vector<ll> deltas={u, 3*u, 9*u, 27*u};

    vector<int> g(NN,0);           // current stable grid
    vector<ll> a(K,0);
    ll F = score(g), used = 0;
    const int MAXIT = 4000;

    for(int it=0; it<MAXIT; it++){
        ll bestGain=0, bestD=0; int bestS=-1;
        vector<int> bestG;
        for(int s=0;s<K;s++){
            for(ll d : deltas){
                if(used + d > B) continue;
                vector<int> tmp = g;
                tmp[src[s]] += (int)d;
                stabilize(tmp, sink);
                ll nF = score(tmp);
                ll gain = nF - F;
                if(gain > bestGain){ bestGain=gain; bestS=s; bestD=d; bestG.swap(tmp); }
            }
        }
        if(bestS < 0) break;       // no improving superposition move
        g.swap(bestG);
        F += bestGain; used += bestD; a[bestS] += bestD;
    }

    for(int i=0;i<K;i++) printf("%lld%c", a[i], i+1<K?' ':'\n');
    return 0;
}
