// TIER: trivial
// Place the single best-value regular tiling (matches the checker's baseline B exactly).
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
  int best=-1; long long bestVal=-1; int bw=0,bh=0; long long btiles=0;
  for(int i=0;i<N;i++){
    int mxx=0,mxy=0; for(auto&c:shape[i]){mxx=max(mxx,c.first);mxy=max(mxy,c.second);}
    int w=mxx+1,h=mxy+1;
    long long tiles=(long long)(W/w)*(long long)(H/h);
    tiles=min(tiles,(long long)cap[i]);
    long long value=tiles*val[i];
    if(value>bestVal){bestVal=value;best=i;bw=w;bh=h;btiles=tiles;}
  }
  int cols=W/bw, rows=H/bh;
  vector<tuple<int,int,int,int>> out;
  long long placed=0;
  for(int r=0;r<rows&&placed<btiles;r++)
    for(int c=0;c<cols&&placed<btiles;c++){
      out.push_back({best+1,0,c*bw,r*bh}); placed++;
    }
  printf("%d\n",(int)out.size());
  for(auto&o:out) printf("%d %d %d %d\n",get<0>(o),get<1>(o),get<2>(o),get<3>(o));
  return 0;
}
