// TIER: strong
// Insight: this is inverse morphogenesis. Don't store the mural in the seed --
// find a RULE whose dynamics generate it, then optimize a sparse seed by
// forward-simulation search so the pattern self-organizes (covering cells outside
// the box that no seeding strategy can reach).
//   1. screen a LIBRARY of outer-totalistic rules (incl. replicator/seeds/gnarl/
//      34-life/diamoeba, plus the famous ones so strong >= greedy by construction);
//   2. on the best rules run greedy forward-selection of seed cells + local toggles,
//      using overlap F(=TP-FP) as the fitness.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N,K,T,r0,c0,b;
vector<uint8_t> tgt;
long long simBudget = 90000; // hard cap on full simulations (keeps well under TL)
long long simUsed = 0;

static inline int MK(initializer_list<int> cs){int m=0;for(int c:cs)m|=(1<<c);return m;}

ll evalGrid(const vector<uint8_t>& seedGrid,int Bmask,int Smask){
    simUsed++;
    static vector<uint8_t> g,h;
    g=seedGrid; h.assign(N*N,0);
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

    // ---- rule library (superset of greedy's famous rules) ----
    vector<pair<int,int>> lib = {
        { MK({}),        MK({0,1,2,3,4,5,6,7,8}) }, // 0 identity
        { MK({3}),       MK({2,3}) },               // 1 Conway
        { MK({3}),       MK({0,1,2,3,4,5,6,7,8}) }, // 2 fill / LwoD
        { MK({2}),       MK({}) },                  // 3 Seeds
        { MK({1,3,5,7}), MK({1,3,5,7}) },           // 4 Replicator
        { MK({1}),       MK({1}) },                 // 5 gnarl
        { MK({3,4}),     MK({3,4}) },               // 6 34-Life
        { MK({3,6}),     MK({2,3}) },               // 7 HighLife
        { MK({3,5,6,7,8}),MK({5,6,7,8}) },          // 8 Diamoeba
        { MK({3,6,7,8}), MK({3,4,6,7,8}) },         // 9 Day&Night
        { MK({2,3}),     MK({}) },                  // 10 B23/S
        { MK({3,4,5}),   MK({5}) },                 // 11 B345/S5
        { MK({2,3,4}),   MK({}) },                  // 12 B234/S
        { MK({3,5,6,7,8}),MK({2,3,4,5}) },          // 13 Coagulations-ish
        { MK({3}),       MK({4,5,6,7,8}) }          // 14 Coral
    };
    int L = lib.size();

    // box target cells sorted by local target density (greedy's seed order)
    vector<tuple<int,int,int>> tc;
    for(int r=r0;r<r0+b;r++)for(int c=c0;c<c0+b;c++) if(tgt[r*N+c]){
        int d=0;
        for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
            int nr=r+dr,nc=c+dc; if(nr<0||nr>=N||nc<0||nc>=N)continue;
            if(tgt[nr*N+nc]) d++;
        }
        tc.push_back({-d,r,c});
    }
    sort(tc.begin(),tc.end());
    vector<pair<int,int>> greedySeed;
    for(auto&t:tc){ if((int)greedySeed.size()>=K)break; greedySeed.push_back({get<1>(t),get<2>(t)}); }
    auto toGrid=[&](const vector<pair<int,int>>& s){
        vector<uint8_t> g(N*N,0); for(auto&p:s) g[p.first*N+p.second]=1; return g;
    };
    vector<uint8_t> greedyGrid=toGrid(greedySeed);

    // ---- screen every rule with the greedy seed (this dominates greedy) ----
    vector<pair<ll,int>> screen;
    ll bestF=-1; int bestB=0,bestS=0; vector<pair<int,int>> bestSeed=greedySeed;
    for(int i=0;i<L;i++){
        ll F=evalGrid(greedyGrid,lib[i].first,lib[i].second);
        screen.push_back({F,i});
        if(F>bestF){bestF=F;bestB=lib[i].first;bestS=lib[i].second;bestSeed=greedySeed;}
    }
    sort(screen.rbegin(),screen.rend());

    // box cell list
    vector<pair<int,int>> box;
    for(int r=r0;r<r0+b;r++)for(int c=c0;c<c0+b;c++) box.push_back({r,c});

    // ---- forward-selection + local toggle on the top rules ----
    int topRules = min(4, L);
    for(int ti=0; ti<topRules && simUsed < simBudget; ti++){
        int Bmask=lib[screen[ti].second].first, Smask=lib[screen[ti].second].second;

        // forward selection from empty seed
        vector<uint8_t> cur(N*N,0);
        int placed=0; ll curF=evalGrid(cur,Bmask,Smask);
        while(placed<K && simUsed<simBudget){
            ll gainBest=-1; int gr=-1,gc=-1;
            for(auto&cell:box){
                int idx=cell.first*N+cell.second;
                if(cur[idx])continue;
                cur[idx]=1; ll F=evalGrid(cur,Bmask,Smask); cur[idx]=0;
                if(F>gainBest){gainBest=F;gr=cell.first;gc=cell.second;}
                if(simUsed>=simBudget)break;
            }
            if(gr<0) break;
            if(gainBest<=curF) break;           // no improving cell
            cur[gr*N+gc]=1; curF=gainBest; placed++;
            if(curF>bestF){
                bestF=curF;bestB=Bmask;bestS=Smask;
                bestSeed.clear();
                for(auto&cell:box) if(cur[cell.first*N+cell.second]) bestSeed.push_back(cell);
            }
        }

        // local toggle pass: try flipping each box cell (respect budget K)
        bool improved=true; int passes=0;
        while(improved && passes<3 && simUsed<simBudget){
            improved=false; passes++;
            for(auto&cell:box){
                if(simUsed>=simBudget)break;
                int idx=cell.first*N+cell.second;
                int live=0; for(auto&c2:box) live+=cur[c2.first*N+c2.second];
                if(cur[idx]==0 && live>=K) continue;
                cur[idx]^=1; ll F=evalGrid(cur,Bmask,Smask);
                if(F>curF){ curF=F; improved=true;
                    if(curF>bestF){ bestF=curF;bestB=Bmask;bestS=Smask;
                        bestSeed.clear();
                        for(auto&c2:box) if(cur[c2.first*N+c2.second]) bestSeed.push_back(c2);
                    }
                } else cur[idx]^=1; // revert
            }
        }
    }

    // ---- also try greedy seed under top rules with a local toggle (safety) ----
    for(int ti=0; ti<min(2,L) && simUsed<simBudget; ti++){
        int Bmask=lib[screen[ti].second].first, Smask=lib[screen[ti].second].second;
        vector<uint8_t> cur=greedyGrid; ll curF=evalGrid(cur,Bmask,Smask);
        for(auto&cell:box){
            if(simUsed>=simBudget)break;
            int idx=cell.first*N+cell.second;
            int live=0; for(auto&c2:box) live+=cur[c2.first*N+c2.second];
            if(cur[idx]==0 && live>=K) continue;
            cur[idx]^=1; ll F=evalGrid(cur,Bmask,Smask);
            if(F>curF){ curF=F;
                if(curF>bestF){ bestF=curF;bestB=Bmask;bestS=Smask;
                    bestSeed.clear();
                    for(auto&c2:box) if(cur[c2.first*N+c2.second]) bestSeed.push_back(c2);
                }
            } else cur[idx]^=1;
        }
    }

    if((int)bestSeed.size()>K) bestSeed.resize(K);
    for(int j=0;j<=8;j++) cout<<((bestB>>j)&1)<<" ";
    for(int j=0;j<=8;j++) cout<<((bestS>>j)&1)<<(j<8?' ':'\n');
    cout<<bestSeed.size()<<"\n";
    for(auto&p:bestSeed) cout<<p.first<<" "<<p.second<<"\n";
    return 0;
}
