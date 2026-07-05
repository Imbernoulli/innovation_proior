// TIER: greedy
// Area greedy: reading-order fill using ALL patch types (orientation 0), covering
// more of the board than the reference but ignoring cell values.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int H,W,T; scanf("%d %d %d",&H,&W,&T);
    vector<vector<int>> v(H, vector<int>(W));
    for(int r=0;r<H;r++)for(int c=0;c<W;c++)scanf("%d",&v[r][c]);
    vector<vector<pair<int,int>>> sh(T);
    vector<int> cnt(T);
    for(int i=0;i<T;i++){int s;scanf("%d %d",&cnt[i],&s);for(int k=0;k<s;k++){int a,b;scanf("%d %d",&a,&b);sh[i].push_back({a,b});}}
    vector<vector<char>> occ(H, vector<char>(W,0));
    vector<array<int,4>> out;
    for(int i=0;i<T;i++){
        auto cs=sh[i];
        int left=cnt[i];
        for(int r=0;r<H && left>0;r++)for(int c=0;c<W && left>0;c++){
            bool fit=true;
            for(auto&p:cs){int ar=r+p.first,ac=c+p.second;if(ar<0||ar>=H||ac<0||ac>=W||occ[ar][ac]){fit=false;break;}}
            if(!fit)continue;
            for(auto&p:cs){occ[r+p.first][c+p.second]=1;}
            out.push_back({i,0,r,c});
            left--;
        }
    }
    printf("%d\n",(int)out.size());
    for(auto&o:out) printf("%d %d %d %d\n",o[0],o[1],o[2],o[3]);
    return 0;
}
