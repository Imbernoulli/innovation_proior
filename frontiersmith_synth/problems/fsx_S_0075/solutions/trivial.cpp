// TIER: trivial
// Reproduces the checker's baseline: anchored greedy with design 0 only.
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
    for(int t=0;t<Pn;t++){
        int c,s; cin>>c>>s; stock[t]=c; ssz[t]=s;
        for(int j=0;j<s;j++){ int dx,dy; cin>>dx>>dy; base[t].push_back({dx,dy}); }
    }
    vector<Shape> o0 = orientations(base[0]);

    vector<vector<char>> cov(H, vector<char>(W,0));
    int used=0;
    vector<pair<int,Shape>> out; // (type, cells)
    for(int cy=0; cy<H && used<stock[0]; cy++){
        for(int cx=0; cx<W && used<stock[0]; cx++){
            if(cov[cy][cx]||vent[cy][cx]) continue;
            for(auto& sh:o0){
                int ax=INT_MAX;
                for(auto&p:sh) if(p.second==0) ax=min(ax,p.first);
                int ox=cx-ax, oy=cy;
                bool fit=true;
                for(auto&p:sh){ int X=p.first+ox,Y=p.second+oy; if(X<0||X>=W||Y<0||Y>=H||vent[Y][X]||cov[Y][X]){fit=false;break;} }
                if(fit){
                    Shape cs;
                    for(auto&p:sh){ int X=p.first+ox,Y=p.second+oy; cov[Y][X]=1; cs.push_back({X,Y}); }
                    out.push_back({0,cs}); used++;
                    break;
                }
            }
        }
    }
    printf("%d\n", (int)out.size());
    for(auto&pr:out){
        printf("%d", pr.first);
        for(auto&c:pr.second) printf(" %d %d", c.first, c.second);
        printf("\n");
    }
    return 0;
}
