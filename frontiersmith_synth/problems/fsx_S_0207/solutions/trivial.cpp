// TIER: trivial
// Replicates the judge's reference construction exactly: reading-order fill with
// type 0 (the 2x2 square) in orientation 0, up to cnt[0] copies. => ratio ~ 0.1
#include <bits/stdc++.h>
using namespace std;
int main(){
    int H,W,T; scanf("%d %d %d",&H,&W,&T);
    vector<vector<int>> v(H, vector<int>(W));
    for(int r=0;r<H;r++)for(int c=0;c<W;c++)scanf("%d",&v[r][c]);
    vector<vector<pair<int,int>>> sh(T);
    vector<int> cnt(T);
    for(int i=0;i<T;i++){int s;scanf("%d %d",&cnt[i],&s);for(int k=0;k<s;k++){int a,b;scanf("%d %d",&a,&b);sh[i].push_back({a,b});}}
    auto cs = sh[0];
    vector<vector<char>> occ(H, vector<char>(W,0));
    vector<array<int,4>> out;
    int left=cnt[0];
    for(int r=0;r<H && left>0;r++)for(int c=0;c<W && left>0;c++){
        bool fit=true;
        for(auto&p:cs){int ar=r+p.first,ac=c+p.second;if(ar<0||ar>=H||ac<0||ac>=W||occ[ar][ac]){fit=false;break;}}
        if(!fit)continue;
        for(auto&p:cs){occ[r+p.first][c+p.second]=1;}
        out.push_back({0,0,r,c});
        left--;
    }
    printf("%d\n",(int)out.size());
    for(auto&o:out) printf("%d %d %d %d\n",o[0],o[1],o[2],o[3]);
    return 0;
}
