// TIER: greedy
// The obvious first attempt: TUNE A SINGLE PILE. Try each source on its own with a few
// drop sizes, stabilize, and keep the one (source, amount) that scores best. This can
// only ever reproduce one localized avalanche lobe; it never combines sources, so on a
// target built from several spread-out drops it captures one region and misses the rest.
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

    auto score=[&](vector<int>&g)->ll{
        ll F=0; for(int i=0;i<NN;i++) if(!sink[i]&&g[i]==T[i]) F+=W[i]; return F;
    };

    ll m0 = max((ll)6, (ll)llround(1.2*NN/(double)max(1,K)));
    vector<ll> amounts={0, m0, 2*m0, 4*m0};

    int bestSrc=0; ll bestAmt=0, bestF=-1;
    for(int s=0;s<K;s++){
        for(ll a : amounts){
            if(a>B) continue;
            vector<int> g(NN,0);
            g[src[s]] += (int)a;
            stabilize(g,sink);
            ll F=score(g);
            if(F>bestF){bestF=F;bestSrc=s;bestAmt=a;}
        }
    }

    for(int i=0;i<K;i++) printf("%lld%c", (i==bestSrc?bestAmt:0LL), i+1<K?' ':'\n');
    return 0;
}
