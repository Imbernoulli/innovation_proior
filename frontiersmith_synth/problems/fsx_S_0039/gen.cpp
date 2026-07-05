#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;

// generate a connected polyomino of ~area cells with bbox side <= cap
vector<pii> genPoly(int area, int cap){
  set<pii> cells;
  cells.insert({0,0});
  int dx[4]={1,-1,0,0}, dy[4]={0,0,1,-1};
  while((int)cells.size()<area){
    // candidate neighbours keeping bbox <= cap
    set<pii> cand;
    for(auto&c:cells){
      for(int d=0;d<4;d++){
        pii np={c.first+dx[d], c.second+dy[d]};
        if(cells.count(np)) continue;
        int mnx=np.first,mxx=np.first,mny=np.second,mxy=np.second;
        for(auto&e:cells){mnx=min(mnx,e.first);mxx=max(mxx,e.first);mny=min(mny,e.second);mxy=max(mxy,e.second);}
        if(mxx-mnx>cap-1) continue;
        if(mxy-mny>cap-1) continue;
        cand.insert(np);
      }
    }
    if(cand.empty()) break;
    vector<pii> v(cand.begin(),cand.end());
    cells.insert(v[rnd.next((int)v.size())]);
  }
  // normalise
  int mnx=INT_MAX,mny=INT_MAX;
  for(auto&c:cells){mnx=min(mnx,c.first);mny=min(mny,c.second);}
  vector<pii> out;
  for(auto&c:cells) out.push_back({c.first-mnx,c.second-mny});
  sort(out.begin(),out.end());
  return out;
}

int main(int argc,char*argv[]){
  registerGen(argc,argv,1);
  int t=atoi(argv[1]);
  int W,H,N;
  if(t<=1){ W=5; H=5; N=3; }
  else {
    W = min(28, 6 + 2*t);
    H = min(28, 5 + 2*t);
    N = min(12, 2 + t);
  }
  int cap = min(4, min(W,H)-1);
  if(cap<1) cap=1;

  printf("%d %d %d\n", W, H, N);
  for(int i=0;i<N;i++){
    int area = rnd.next(2, min(8, cap*cap));
    if(area<1) area=1;
    vector<pii> poly = genPoly(area, cap);
    int m = (int)poly.size();
    int dens = rnd.next(8, 12);       // narrow density band -> no score capping
    int v = m * dens;
    if(v>200) v=200;
    if(v<1) v=1;
    int c = rnd.next(3, 12);
    printf("%d %d %d\n", m, v, c);
    for(auto&p:poly) printf("%d %d\n", p.first, p.second);
  }
  return 0;
}
