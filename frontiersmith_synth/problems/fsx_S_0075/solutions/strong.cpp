// TIER: strong
// Global-best greedy: each round, deploy the single highest-yield feasible pod anywhere.
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> P;
typedef vector<P> Shape;

static Shape norm(Shape s){
    int mx=INT_MAX,my=INT_MAX;
    for(auto&p:s){mx=min(mx,p.first);my=min(my,p.second);}
    for(auto&p:s){p.first-=mx;p.second-=my;}
    sort(s.begin(),s.end());
    return s;
}
static vector<Shape> orientations(const Shape& base){
    set<Shape> res;
    Shape cur=base;
    for(int f=0;f<2;f++){
        Shape c=cur;
        for(int r=0;r<4;r++){
            res.insert(norm(c));
            Shape nx; for(auto&p:c) nx.push_back({p.second,-p.first});
            c=nx;
        }
        Shape rf; for(auto&p:cur) rf.push_back({-p.first,p.second});
        cur=rf;
    }
    return vector<Shape>(res.begin(),res.end());
}

int main(){
    int W,H;
    if(!(cin>>W>>H)) return 0;
    vector<vector<int>> val(H, vector<int>(W));
    for(int y=0;y<H;y++) for(int x=0;x<W;x++) cin>>val[y][x];
    int K; cin>>K;
    vector<vector<char>> vent(H, vector<char>(W,0));
    for(int i=0;i<K;i++){ int ox,oy; cin>>ox>>oy; vent[oy][ox]=1; }
    int Pn; cin>>Pn;
    vector<Shape> base(Pn); vector<int> stock(Pn), ssz(Pn);
    vector<vector<Shape>> oris(Pn);
    for(int t=0;t<Pn;t++){
        int c,s; cin>>c>>s; stock[t]=c; ssz[t]=s;
        for(int j=0;j<s;j++){ int dx,dy; cin>>dx>>dy; base[t].push_back({dx,dy}); }
        oris[t]=orientations(base[t]);
    }

    vector<vector<char>> cov(H, vector<char>(W,0));
    vector<int> used(Pn,0);
    vector<pair<int,Shape>> out;

    int maxRounds = W*H + 5;
    for(int round=0; round<maxRounds; round++){
        long long bestKey1=-1; double bestKey2=-1; int bestT=-1; Shape bestCells;
        for(int t=0;t<Pn;t++){
            if(used[t]>=stock[t]) continue;
            for(auto& sh:oris[t]){
                for(int oy=0; oy<H; oy++){
                    for(int ox=0; ox<W; ox++){
                        bool fit=true; long long add=0; Shape cs;
                        for(auto&p:sh){
                            int X=p.first+ox, Y=p.second+oy;
                            if(X<0||X>=W||Y<0||Y>=H||vent[Y][X]||cov[Y][X]){fit=false;break;}
                            add+=val[Y][X]; cs.push_back({X,Y});
                        }
                        if(!fit) continue;
                        double dens = (double)add/(double)sh.size();
                        // primary: total yield; tie-break: yield density
                        if(add>bestKey1 || (add==bestKey1 && dens>bestKey2)){
                            bestKey1=add; bestKey2=dens; bestT=t; bestCells=cs;
                        }
                    }
                }
            }
        }
        if(bestT<0) break;
        for(auto&c:bestCells) cov[c.second][c.first]=1;
        out.push_back({bestT,bestCells}); used[bestT]++;
    }
    printf("%d\n", (int)out.size());
    for(auto&pr:out){
        printf("%d", pr.first);
        for(auto&c:pr.second) printf(" %d %d", c.first, c.second);
        printf("\n");
    }
    return 0;
}
