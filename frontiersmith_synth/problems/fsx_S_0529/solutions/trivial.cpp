// TIER: trivial
// Baseline: build exactly the single canonical triangulated ladder in the two
// leftmost columns -- identical to the checker's reference truss, so F == B and the
// score is the calibration point (~0.1).
#include <bits/stdc++.h>
using namespace std;
int main(){
    int H,W; long long Budget;
    scanf("%d %d %lld",&H,&W,&Budget);
    vector<long long> C(H);
    for(int i=0;i<H;i++) scanf("%lld",&C[i]);
    auto nid=[&](int i,int j){ return i*(W+1)+j; };
    vector<pair<int,int>> mem;
    for(int i=0;i<H;i++){
        mem.push_back({nid(i,0),nid(i+1,0)});
        mem.push_back({nid(i,1),nid(i+1,1)});
        mem.push_back({nid(i,0),nid(i,1)});
        mem.push_back({nid(i,0),nid(i+1,1)});
    }
    mem.push_back({nid(H,0),nid(H,1)});
    printf("%d\n",(int)mem.size());
    for(auto&m:mem) printf("%d %d\n",m.first,m.second);
    return 0;
}
