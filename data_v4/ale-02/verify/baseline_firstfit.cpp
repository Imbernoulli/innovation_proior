// First-fit greedy baseline for ALE-02 (the reference the score ratio is against).
// Largest-area-first, scan all anchors, place whenever it fits. Always feasible.
#include <bits/stdc++.h>
using namespace std;
int H, W, K;
struct Shape { int hh, ww, area; vector<pair<int,int>> cells; vector<pair<int,uint64_t>> rows; };
vector<array<Shape,4>> shapes;
vector<int> areaOf, cntOf, usedCnt;
vector<uint64_t> occ;
static vector<pair<int,int>> rotate_norm(const vector<pair<int,int>>& cells, int rot){
    vector<pair<int,int>> p=cells;
    for(int t=0;t<(rot&3);t++) for(auto&q:p) q={q.second,-q.first};
    int mr=INT_MAX,mc=INT_MAX; for(auto&q:p){mr=min(mr,q.first);mc=min(mc,q.second);}
    for(auto&q:p){q.first-=mr;q.second-=mc;} sort(p.begin(),p.end());
    p.erase(unique(p.begin(),p.end()),p.end()); return p;
}
static inline bool fits(const Shape&s,int ar,int ac){
    if(ar<0||ac<0||ar+s.hh>H||ac+s.ww>W) return false;
    for(auto&pr:s.rows){ if(occ[ar+pr.first]&(pr.second<<ac)) return false; } return true;
}
int main(){
    if(scanf("%d %d %d",&H,&W,&K)!=3) return 0;
    areaOf.assign(K,0); cntOf.assign(K,0); shapes.assign(K,{}); usedCnt.assign(K,0);
    vector<vector<pair<int,int>>> base(K);
    for(int k=0;k<K;k++){ int A,cnt; scanf("%d %d",&A,&cnt); areaOf[k]=A; cntOf[k]=cnt;
        base[k].resize(A); for(int i=0;i<A;i++) scanf("%d %d",&base[k][i].first,&base[k][i].second); }
    for(int k=0;k<K;k++) for(int rot=0;rot<4;rot++){
        auto cells=rotate_norm(base[k],rot); Shape s; s.cells=cells; s.area=(int)cells.size();
        int hh=0,ww=0; for(auto&c:cells){hh=max(hh,c.first+1);ww=max(ww,c.second+1);} s.hh=hh; s.ww=ww;
        map<int,uint64_t> mp; for(auto&c:cells) mp[c.first]|=(uint64_t)1<<c.second;
        for(auto&pr:mp) s.rows.push_back(pr); shapes[k][rot]=move(s);
    }
    occ.assign(H,0);
    vector<array<int,4>> out; // k rot ar ac
    vector<int> order(K); iota(order.begin(),order.end(),0);
    sort(order.begin(),order.end(),[&](int a,int b){return areaOf[a]>areaOf[b];});
    for(int k:order) for(int rot=0;rot<4;rot++){
        const Shape&s=shapes[k][rot]; bool dup=false;
        for(int r2=0;r2<rot;r2++) if(shapes[k][r2].cells==s.cells){dup=true;break;}
        if(dup) continue;
        for(int ar=0;ar+s.hh<=H&&usedCnt[k]<cntOf[k];ar++)
            for(int ac=0;ac+s.ww<=W&&usedCnt[k]<cntOf[k];ac++)
                if(fits(s,ar,ac)){ for(auto&pr:s.rows) occ[ar+pr.first]^=(pr.second<<ac);
                    usedCnt[k]++; out.push_back({k,rot,ar,ac}); }
    }
    printf("%d\n",(int)out.size());
    for(auto&o:out) printf("%d %d %d %d\n",o[0],o[1],o[2],o[3]);
    return 0;
}
