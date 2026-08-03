// TIER: invalid
// Rope off EVERY doorway: exceeds the closure budget and disconnects the museum
// -> the checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n,m,s,t; long long K;
    if(!(cin>>n>>m>>s>>t>>K)) return 0;
    for(int i=0;i<m;i++){int u,v; long long w,c; cin>>u>>v>>w>>c;}
    printf("%d\n",m);
    for(int i=1;i<=m;i++) printf("%d\n",i);
    return 0;
}
