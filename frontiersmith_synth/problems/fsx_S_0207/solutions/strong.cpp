// TIER: strong
// Value-density greedy: enumerate every (type, orientation, offset) candidate,
// rank by covered conservation value, and place first-fit under supply/overlap
// constraints. Targets the high-value clusters instead of just area.
#include <bits/stdc++.h>
using namespace std;

int H,W,T;
vector<vector<int>> v;
vector<vector<pair<int,int>>> sh;
vector<int> cnt;

vector<pair<int,int>> oriented(int i,int o){
    int reflect=o/4, rot=o%4;
    vector<pair<int,int>> cs;
    for(auto&p:sh[i]){
        int x=p.first,y=p.second;
        if(reflect)y=-y;
        for(int k=0;k<rot;k++){int nx=y,ny=-x;x=nx;y=ny;}
        cs.push_back({x,y});
    }
    int mr=INT_MAX,mc=INT_MAX;
    for(auto&p:cs){mr=min(mr,p.first);mc=min(mc,p.second);}
    for(auto&p:cs){p.first-=mr;p.second-=mc;}
    sort(cs.begin(),cs.end());
    return cs;
}

int main(){
    scanf("%d %d %d",&H,&W,&T);
    v.assign(H, vector<int>(W));
    for(int r=0;r<H;r++)for(int c=0;c<W;c++)scanf("%d",&v[r][c]);
    sh.resize(T); cnt.resize(T);
    for(int i=0;i<T;i++){int s;scanf("%d %d",&cnt[i],&s);for(int k=0;k<s;k++){int a,b;scanf("%d %d",&a,&b);sh[i].push_back({a,b});}}

    struct Cand{ long long val; int i,o,r,c; };
    vector<Cand> cand;
    for(int i=0;i<T;i++){
        set<vector<pair<int,int>>> seen;
        for(int o=0;o<8;o++){
            auto cs=oriented(i,o);
            if(!seen.insert(cs).second) continue;
            int mh=0,mw=0; for(auto&p:cs){mh=max(mh,p.first);mw=max(mw,p.second);}
            for(int r=0;r+mh<H+1 && r<H;r++) for(int c=0;c+mw<W+1 && c<W;c++){
                if(r+mh>=H||c+mw>=W) continue;
                long long val=0;
                for(auto&p:cs) val+=v[r+p.first][c+p.second];
                cand.push_back({val,i,o,r,c});
            }
        }
    }
    sort(cand.begin(),cand.end(),[](const Cand&a,const Cand&b){return a.val>b.val;});

    vector<vector<char>> occ(H, vector<char>(W,0));
    vector<int> used(T,0);
    vector<array<int,4>> out;
    for(auto&cd:cand){
        if(used[cd.i]>=cnt[cd.i]) continue;
        auto cs=oriented(cd.i,cd.o);
        bool fit=true;
        for(auto&p:cs){int ar=cd.r+p.first,ac=cd.c+p.second;if(occ[ar][ac]){fit=false;break;}}
        if(!fit)continue;
        for(auto&p:cs) occ[cd.r+p.first][cd.c+p.second]=1;
        used[cd.i]++;
        out.push_back({cd.i,cd.o,cd.r,cd.c});
    }
    printf("%d\n",(int)out.size());
    for(auto&o:out) printf("%d %d %d %d\n",o[0],o[1],o[2],o[3]);
    return 0;
}
