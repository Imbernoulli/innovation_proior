// TIER: greedy
// The obvious approach: every machine uses Shortest-Processing-Time-first (A_i=1, B_i=0),
// the textbook "good" coordination mechanism for scheduling games. It looks unbeatable --
// each machine locally optimizes its own total completion time -- but it never uses the job
// weight, and it lets any individually-fast job assume it can always claim a good spot near
// the front of a popular machine's queue, regardless of how many similar jobs are already
// piling in there. On instances where many jobs are near-tied for the same machine, that
// self-reinforcing over-subscription lands the equilibrium far from balanced.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n,m;
    if(scanf("%d %d",&n,&m)!=2) return 0;
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){ long long x; scanf("%lld",&x); }
        long long ww; scanf("%lld",&ww);
    }
    for(int i=0;i<m;i++) printf("1 0\n");
    return 0;
}
