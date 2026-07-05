#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> P;
typedef vector<P> Shape;

static Shape norm(Shape s){
    int mx=INT_MAX,my=INT_MAX;
    for(auto&p:s){mx=min(mx,p.first);my=min(my,p.second);}
    for(auto&p:s){p.first-=mx;p.second-=my;}
    sort(s.begin(),s.end());
    return s;
}

// all distinct orientations (rotations x reflections), normalized, sorted set
static vector<Shape> orientations(const Shape& base){
    set<Shape> res;
    Shape cur=base;
    for(int f=0;f<2;f++){
        Shape c=cur;
        for(int r=0;r<4;r++){
            res.insert(norm(c));
            Shape nx; for(auto&p:c) nx.push_back({p.second,-p.first}); // rotate 90
            c=nx;
        }
        Shape rf; for(auto&p:cur) rf.push_back({-p.first,p.second}); // reflect x
        cur=rf;
    }
    return vector<Shape>(res.begin(),res.end());
}

int main(int argc, char** argv){
    registerTestlibCmd(argc, argv);

    int W = inf.readInt();
    int H = inf.readInt();
    vector<vector<int>> val(H, vector<int>(W));
    for(int y=0;y<H;y++) for(int x=0;x<W;x++) val[y][x]=inf.readInt();
    int K = inf.readInt();
    vector<vector<char>> vent(H, vector<char>(W,0));
    for(int i=0;i<K;i++){ int ox=inf.readInt(), oy=inf.readInt(); vent[oy][ox]=1; }
    int Pn = inf.readInt();
    vector<Shape> base(Pn);
    vector<int> stock(Pn), ssz(Pn);
    vector<vector<Shape>> oris(Pn);
    vector<set<Shape>> oriset(Pn);
    for(int t=0;t<Pn;t++){
        int c=inf.readInt(), s=inf.readInt();
        stock[t]=c; ssz[t]=s;
        for(int j=0;j<s;j++){ int dx=inf.readInt(), dy=inf.readInt(); base[t].push_back({dx,dy}); }
        oris[t]=orientations(base[t]);
        oriset[t]=set<Shape>(oris[t].begin(), oris[t].end());
    }

    // ---- internal baseline B: anchored greedy with design 0 only ----
    auto computeB=[&]()->ll{
        vector<vector<char>> cov(H, vector<char>(W,0));
        int used=0;
        ll harvest=0;
        for(int cy=0; cy<H && used<stock[0]; cy++){
            for(int cx=0; cx<W && used<stock[0]; cx++){
                if(cov[cy][cx] || vent[cy][cx]) continue;
                // try orientations in sorted order; anchor = topmost then leftmost
                bool placed=false;
                for(auto& sh : oris[0]){
                    int ax=INT_MAX;
                    for(auto&p:sh) if(p.second==0) ax=min(ax,p.first);
                    int ox=cx-ax, oy=cy-0;
                    bool fit=true;
                    for(auto&p:sh){
                        int X=p.first+ox, Y=p.second+oy;
                        if(X<0||X>=W||Y<0||Y>=H||vent[Y][X]||cov[Y][X]){ fit=false; break; }
                    }
                    if(fit){
                        for(auto&p:sh){ int X=p.first+ox, Y=p.second+oy; cov[Y][X]=1; harvest+=val[Y][X]; }
                        used++; placed=true; break;
                    }
                }
                (void)placed;
            }
        }
        return harvest;
    };
    ll B = max((ll)1, computeB());

    // ---- read & validate participant ----
    vector<vector<char>> covered(H, vector<char>(W,0));
    vector<int> usedT(Pn,0);
    int M = ouf.readInt(0, W*H+5, "M");
    for(int i=0;i<M;i++){
        int t = ouf.readInt(0, Pn-1, "t");
        int s = ssz[t];
        Shape cells;
        set<P> seen;
        for(int j=0;j<s;j++){
            int x = ouf.readInt(0, W-1, "x");
            int y = ouf.readInt(0, H-1, "y");
            if(seen.count({x,y})) quitf(_wa, "pod %d: repeated cell (%d,%d)", i, x, y);
            seen.insert({x,y});
            if(vent[y][x]) quitf(_wa, "pod %d: cell (%d,%d) is a vent", i, x, y);
            if(covered[y][x]) quitf(_wa, "pod %d: cell (%d,%d) overlaps another pod", i, x, y);
            cells.push_back({x,y});
        }
        Shape nc = norm(cells);
        if(!oriset[t].count(nc)) quitf(_wa, "pod %d: cells are not a legal copy of design %d", i, t);
        for(auto&p:cells) covered[p.second][p.first]=1;
        usedT[t]++;
        if(usedT[t] > stock[t]) quitf(_wa, "design %d: used %d copies > stock %d", t, usedT[t], stock[t]);
    }
    if(!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F=0;
    for(int y=0;y<H;y++) for(int x=0;x<W;x++) if(covered[y][x]) F+=val[y][x];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc/1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc/1000.0);
    return 0;
}
