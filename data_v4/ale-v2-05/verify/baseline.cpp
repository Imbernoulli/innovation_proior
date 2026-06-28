// Trivial baseline for ale-v2-05 (UFLP): open ALL facilities.
// Always feasible (M = F >= 1, distinct indices). Service cost is minimal but
// it pays every opening cost, so it is a weak-but-legal reference to beat.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int F, C;
    if(!(cin>>F>>C)) return 0;
    for(int i=0;i<F;i++){long long x,y,c;cin>>x>>y>>c;}
    for(int i=0;i<C;i++){long long x,y;cin>>x>>y;}
    cout<<F<<"\n";
    for(int i=0;i<F;i++) cout<<(i+1)<<"\n";
    return 0;
}
