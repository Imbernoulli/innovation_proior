// TIER: invalid
// Deliberately infeasible: release request 1 before ever capturing it.
// Precedence is violated, so the checker must score this 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int P; long long Q;
    if(!(cin>>P>>Q)) return 0;
    long long x0,y0; cin>>x0>>y0;
    for(int i=0;i<P;i++){ long long a,b,c,d,q,w; cin>>a>>b>>c>>d>>q>>w; }
    // one event: release (type 1) request 1 with no prior capture -> infeasible
    printf("1\n");
    printf("1 1\n");
    return 0;
}
