// TIER: trivial
// Close nothing -> shortest tour unchanged -> scores the calibration baseline.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n,m,s,t; long long K;
    if(!(cin>>n>>m>>s>>t>>K)) return 0;
    for(int i=0;i<m;i++){int u,v; long long w,c; cin>>u>>v>>w>>c;}
    printf("0\n");
    return 0;
}
