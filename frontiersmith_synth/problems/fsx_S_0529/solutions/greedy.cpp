// TIER: greedy
// The obvious heuristic: cover the whole span with a fully triangulated single-diagonal
// sheet, then "make it stronger" by X-bracing panels in reading order (top level first)
// until the steel runs out.  It triangulates correctly but pours its spare steel into
// whichever level it reaches first -- blind to WHICH stratum is the real bottleneck.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int H,W; long long Budget;
    scanf("%d %d %lld",&H,&W,&Budget);
    vector<long long> C(H);
    for(int i=0;i<H;i++) scanf("%lld",&C[i]);
    auto nid=[&](int i,int j){ return i*(W+1)+j; };
    vector<pair<int,int>> mem;
    long long steel=0;
    auto add=[&](int a,int b){
        long long dx=(a%(W+1))-(b%(W+1)), dy=(a/(W+1))-(b/(W+1));
        mem.push_back({a,b}); steel+=dx*dx+dy*dy;
    };
    // full single-diagonal sheet
    for(int j=0;j<=W;j++) for(int i=0;i<H;i++) add(nid(i,j),nid(i+1,j));   // verticals
    for(int i=0;i<=H;i++) for(int j=0;j<W;j++) add(nid(i,j),nid(i,j+1));   // rungs
    for(int i=0;i<H;i++)  for(int j=0;j<W;j++) add(nid(i,j),nid(i+1,j+1)); // diagonals
    // X-brace in reading order until budget out
    for(int i=0;i<H;i++) for(int j=0;j<W;j++){
        if(steel+2<=Budget) add(nid(i,j+1),nid(i+1,j));
    }
    printf("%d\n",(int)mem.size());
    for(auto&m:mem) printf("%d %d\n",m.first,m.second);
    return 0;
}
