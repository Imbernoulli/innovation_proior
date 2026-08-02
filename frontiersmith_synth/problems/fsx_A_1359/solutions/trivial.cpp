// TIER: trivial
// Longest-processing-time-first on every machine (A_i=-1, B_i=0) -- the checker's own
// reference construction, ignoring job weight entirely. Do-nothing baseline.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n,m;
    if(scanf("%d %d",&n,&m)!=2) return 0;
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){ long long x; scanf("%lld",&x); }
        long long ww; scanf("%lld",&ww);
    }
    for(int i=0;i<m;i++) printf("-1 0\n");
    return 0;
}
