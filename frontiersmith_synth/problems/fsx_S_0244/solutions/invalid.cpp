// TIER: invalid
// Deliberately infeasible: delivers request 1 without ever picking it up
// (violates precedence / empty-stack LIFO). Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int P,H; long long x0,y0;
    if(!(cin>>P>>H)) return 0;
    cin>>x0>>y0;
    long long a,b,c,d,w;
    for(int i=0;i<P;i++) cin>>a>>b>>c>>d>>w;
    printf("1\n");
    printf("1 1\n");   // deliver request 1 with no prior pickup -> infeasible
    return 0;
}
