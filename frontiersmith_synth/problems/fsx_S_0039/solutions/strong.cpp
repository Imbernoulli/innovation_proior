// TIER: strong
// Fill-first-free-cell packing with all 8 orientations + seeded multi-restart; keep best revenue.
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;

int W,H,N;
vector<long long> val; vector<int> cap;
vector<array<vector<pii>,8>> osh; // 8 oriented (normalised) shapes per type

vector<pii> orient(const vector<pii>& cells,int r){
  vector<pii> out; out.reserve(cells.size());
  for(auto&c:cells){int x=c.first,y=c.second,nx=0,ny=0;
    switch(r){case 0:nx=x;ny=y;break;case 1:nx=-y;ny=x;break;case 2:nx=-x;ny=-y;break;
      case 3:nx=y;ny=-x;break;case 4:nx=-x;ny=y;break;case 5:nx=y;ny=x;break;
      case 6:nx=x;ny=-y;break;case 7:nx=-y;ny=-x;break;}
    out.push_back({nx,ny});}
  int mnx=INT_MAX,mny=INT_MAX; for(auto&c:out){mnx=min(mnx,c.first);mny=min(mny,c.second);}
  for(auto&c:out){c.first-=mnx;c.second-=mny;}
  return out;
}

int main(){
  scanf("%d %d %d",&W,&H,&N);
  vector<vector<pii>> shape(N); val.assign(N,0); cap.assign(N,0); osh.resize(N);
  for(int i=0;i<N;i++){int m;long long v;int c;scanf("%d %lld %d",&m,&v,&c);val[i]=v;cap[i]=c;
    for(int j=0;j<m;j++){int x,y;scanf("%d %d",&x,&y);shape[i].push_back({x,y});}
    for(int r=0;r<8;r++) osh[i][r]=orient(shape[i],r);
  }

  long long bestF=-1; vector<tuple<int,int,int,int>> bestOut;
  int restarts=10;
  for(int s=0;s<restarts;s++){
    mt19937 rng(12345u + 6791u*s);
    vector<vector<char>> grid(W, vector<char>(H,0));
    vector<int> used(N,0);
    vector<tuple<int,int,int,int>> out; long long F=0;
    for(int fy=0;fy<H;fy++) for(int fx=0;fx<W;fx++){
      if(grid[fx][fy]) continue;
      // best placement covering (fx,fy)
      long long bv=-1; unsigned bkey=0; int bi=-1,br=-1,box=-1,boy=-1;
      for(int i=0;i<N;i++){
        if(used[i]>=cap[i]) continue;
        for(int r=0;r<8;r++){
          const auto& cells=osh[i][r];
          int mxx=0,mxy=0; for(auto&c:cells){mxx=max(mxx,c.first);mxy=max(mxy,c.second);}
          int w=mxx+1,h=mxy+1;
          for(auto&a:cells){
            int ox=fx-a.first, oy=fy-a.second;
            if(ox<0||oy<0||ox+w>W||oy+h>H) continue;
            bool ok=true;
            for(auto&c:cells){if(grid[ox+c.first][oy+c.second]){ok=false;break;}}
            if(!ok) continue;
            unsigned key=rng();
            if(val[i]>bv || (val[i]==bv && key>bkey)){
              bv=val[i];bkey=key;bi=i;br=r;box=ox;boy=oy;
            }
          }
        }
      }
      if(bi<0) continue; // cannot fill this cell
      for(auto&c:osh[bi][br]) grid[box+c.first][boy+c.second]=1;
      out.push_back({bi+1,br,box,boy}); used[bi]++; F+=val[bi];
    }
    if(F>bestF){bestF=F;bestOut=out;}
  }
  printf("%d\n",(int)bestOut.size());
  for(auto&o:bestOut) printf("%d %d %d %d\n",get<0>(o),get<1>(o),get<2>(o),get<3>(o));
  return 0;
}
