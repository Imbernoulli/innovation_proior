// TIER: greedy
// The obvious approach: seed the densest target pixels inside the box, then try a
// few FAMOUS rules {identity, Conway B3/S23, fill B3/S012345678} and keep the best.
// It fills blobby regions a little but its fixed rules cannot reproduce
// exotic-dynamics targets, and it never optimizes the seed.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N,K,T,r0,c0,b;
vector<uint8_t> tgt;

static inline int MK(initializer_list<int> cs){int m=0;for(int c:cs)m|=(1<<c);return m;}

ll evalF(const vector<pair<int,int>>& seed,int Bmask,int Smask){
    vector<uint8_t> g(N*N,0),h(N*N,0);
    for(auto&p:seed) g[p.first*N+p.second]=1;
    for(int s=0;s<T;s++){
        for(int r=0;r<N;r++)for(int c=0;c<N;c++){
            int cnt=0;
            for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
                if(!dr&&!dc)continue; int nr=r+dr,nc=c+dc;
                if(nr<0||nr>=N||nc<0||nc>=N)continue; cnt+=g[nr*N+nc];
            }
            int self=g[r*N+c];
            h[r*N+c]=(uint8_t)(self?((Smask>>cnt)&1):((Bmask>>cnt)&1));
        }
        g.swap(h);
    }
    ll TP=0,FP=0;
    for(int i=0;i<N*N;i++) if(g[i]){ if(tgt[i])TP++; else FP++; }
    ll F=TP-FP; return F<0?0:F;
}

int main(){
    if(!(cin>>N>>K>>T>>r0>>c0>>b)) return 0;
    vector<string> gs(N); for(int r=0;r<N;r++) cin>>gs[r];
    tgt.assign(N*N,0);
    for(int r=0;r<N;r++)for(int c=0;c<N;c++) tgt[r*N+c]=(gs[r][c]=='1');

    // seed = top-K box target cells by local target density
    vector<tuple<int,int,int>> cells; // (density, r, c)
    for(int r=r0;r<r0+b;r++)for(int c=c0;c<c0+b;c++) if(tgt[r*N+c]){
        int d=0;
        for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
            int nr=r+dr,nc=c+dc;
            if(nr<0||nr>=N||nc<0||nc>=N)continue;
            if(tgt[nr*N+nc]) d++;
        }
        cells.push_back({-d,r,c});
    }
    sort(cells.begin(),cells.end());
    vector<pair<int,int>> seed;
    for(auto&t:cells){ if((int)seed.size()>=K)break; seed.push_back({get<1>(t),get<2>(t)}); }

    // candidate famous rules
    vector<pair<int,int>> rules = {
        { MK({}), MK({0,1,2,3,4,5,6,7,8}) },   // identity
        { MK({3}), MK({2,3}) },                // Conway
        { MK({3}), MK({0,1,2,3,4,5,6,7,8}) }   // fill / life-without-death
    };

    ll bestF=-1; int bB=0,bS=0;
    for(auto&ru:rules){ ll F=evalF(seed,ru.first,ru.second); if(F>bestF){bestF=F;bB=ru.first;bS=ru.second;} }

    for(int j=0;j<=8;j++) cout<<((bB>>j)&1)<<" ";
    for(int j=0;j<=8;j++) cout<<((bS>>j)&1)<<(j<8?' ':'\n');
    cout<<seed.size()<<"\n";
    for(auto&p:seed) cout<<p.first<<" "<<p.second<<"\n";
    return 0;
}
