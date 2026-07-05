#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;

// apply dihedral transform r then normalise to min 0
vector<pii> orient(const vector<pii>& cells, int r){
  vector<pii> out; out.reserve(cells.size());
  for(auto&c:cells){
    int x=c.first,y=c.second,nx=0,ny=0;
    switch(r){
      case 0: nx= x; ny= y; break;
      case 1: nx=-y; ny= x; break;
      case 2: nx=-x; ny=-y; break;
      case 3: nx= y; ny=-x; break;
      case 4: nx=-x; ny= y; break;
      case 5: nx= y; ny= x; break;
      case 6: nx= x; ny=-y; break;
      case 7: nx=-y; ny=-x; break;
    }
    out.push_back({nx,ny});
  }
  int mnx=INT_MAX,mny=INT_MAX;
  for(auto&c:out){mnx=min(mnx,c.first);mny=min(mny,c.second);}
  for(auto&c:out){c.first-=mnx;c.second-=mny;}
  return out;
}

int main(int argc,char*argv[]){
  registerTestlibCmd(argc,argv);
  int W=inf.readInt(), H=inf.readInt(), N=inf.readInt();
  vector<vector<pii>> shape(N);
  vector<long long> val(N);
  vector<int> cap(N);
  for(int i=0;i<N;i++){
    int m=inf.readInt(); long long v=inf.readInt(); int c=inf.readInt();
    val[i]=v; cap[i]=c;
    for(int j=0;j<m;j++){int x=inf.readInt(),y=inf.readInt();shape[i].push_back({x,y});}
  }
  // baseline B = best single-type axis-aligned tiling with r=0
  long long B=0;
  for(int i=0;i<N;i++){
    int mxx=0,mxy=0;
    for(auto&c:shape[i]){mxx=max(mxx,c.first);mxy=max(mxy,c.second);}
    int w=mxx+1, h=mxy+1;
    long long tiles = (long long)(W/w)*(long long)(H/h);
    tiles = min(tiles, (long long)cap[i]);
    B = max(B, tiles*val[i]);
  }

  // read participant layout
  int K=ouf.readInt(0, W*H, "K");
  vector<vector<char>> grid(W, vector<char>(H,0));
  vector<int> used(N,0);
  long long F=0;
  for(int k=0;k<K;k++){
    int i=ouf.readInt(1,N,"ride")-1;
    int r=ouf.readInt(0,7,"orient");
    int ox=ouf.readInt(0,W-1,"ox");
    int oy=ouf.readInt(0,H-1,"oy");
    used[i]++;
    if(used[i]>cap[i]) quitf(_wa,"ride %d installed more than inventory %d",i+1,cap[i]);
    vector<pii> cells=orient(shape[i], r);
    for(auto&c:cells){
      int x=ox+c.first, y=oy+c.second;
      if(x<0||x>=W||y<0||y>=H) quitf(_wa,"ride %d out of bounds at (%d,%d)",i+1,x,y);
      if(grid[x][y]) quitf(_wa,"overlap at (%d,%d)",x,y);
      grid[x][y]=1;
    }
    F+=val[i];
  }
  if(!ouf.seekEof()) quitf(_wa,"trailing output");
  if(B<=0) B=1;
  double sc=min(1000.0, 100.0*(double)F/(double)max(1LL,B));
  quitp(sc/1000.0,"OK F=%lld B=%lld Ratio: %.6f",F,B,sc/1000.0);
  return 0;
}
