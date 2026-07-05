// TIER: invalid
// Deliberately infeasible: delivers task 1 before ever picking it up. Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    long long Xd,Yd; int m;
    if(!(cin>>Xd>>Yd>>m)) return 0;
    for(int j=0;j<m;j++){long long a,b,c,d,w;cin>>a>>b>>c>>d>>w;}
    printf("1\n");
    printf("1 1\n"); // delivery before pickup -> feasibility violation
    return 0;
}
