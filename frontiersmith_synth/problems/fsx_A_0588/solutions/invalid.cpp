// TIER: invalid
// Deliberately infeasible: transfers far more heat than hot stream 1 can supply
// (over-cools it past its target) -> the checker must reject it -> ratio 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int NH,NC; long long dTmin;
    if(!(cin>>NH>>NC>>dTmin)) return 0;
    long long a,b,cp;
    for(int i=0;i<NH;i++) cin>>a>>b>>cp;
    for(int j=0;j<NC;j++) cin>>a>>b>>cp;
    cout<<1<<"\n";
    cout<<1<<" "<<1<<" "<<(long long)4000000000000LL<<"\n"; // Q >> capH -> infeasible
    return 0;
}
