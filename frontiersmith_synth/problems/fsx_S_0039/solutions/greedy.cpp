// TIER: greedy
// One-pass density-ordered greedy using only orientation r=0.
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;
int main(){
  int W,H,N; scanf("%d %d %d",&W,&H,&N);
  vector<vector<pii>> shape(N); vector<long long> val(N); vector<int> cap(N);
  for(int i=0;i<N;i++){
    int m; long long v; int c; scanf("%d %lld %d",&m,&v,&c);
    val[i]=v; cap[i]=c;
    for(int j=0;j<m;j++){int x,y;scanf("%d %d",&x,&y);shape[i].push_back({x,y});}
  }
  vector<int> order(N); iota(order.begin(),order.end(),0);
  sort(order.begin(),order.end(),[&](int a,int b){
    double da=(double)val[a]/shape[a].size(), db=(double)val[b]/shape[b].size();
    return da>db;
  });
  vector<vector<char>> grid(W, vector<char>(H,0));
  vector<int> used(N,0);
  vector<tuple<int,int,int,int>> out;
  for(int idx=0;idx<N;idx++){
    int i=order[idx];
    int mxx=0,mxy=0; for(auto&c:shape[i]){mxx=max(mxx,c.first);mxy=max(mxy,c.second);}
    int w=mxx+1,h=mxy+1;
    while(used[i]<cap[i]){
      bool placed=false;
      for(int oy=0;oy+h<=H&&!placed;oy++)
        for(int ox=0;ox+w<=W&&!placed;ox++){
          bool ok=true;
          for(auto&c:shape[i]){int x=ox+c.first,y=oy+c.second;if(grid[x][y]){ok=false;break;}}
          if(!ok) continue;
          for(auto&c:shape[i]){grid[ox+c.first][oy+c.second]=1;}
          out.push_back({i+1,0,ox,oy}); used[i]++; placed=true;
        }
      if(!placed) break;
    }
  }
  printf("%d\n",(int)out.size());
  for(auto&o:out) printf("%d %d %d %d\n",get<0>(o),get<1>(o),get<2>(o),get<3>(o));
  return 0;
}
