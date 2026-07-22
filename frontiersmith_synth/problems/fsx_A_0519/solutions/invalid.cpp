// TIER: invalid
// deliberately infeasible: fires every job as "job 1" (not a permutation) and with an
// out-of-window temperature -> checker must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    int N; ll cheat,num,den;
    scanf("%d %lld %lld %lld",&N,&cheat,&num,&den);
    vector<ll> lo(N+1),hi(N+1),d(N+1),D(N+1),w(N+1);
    for(int i=1;i<=N;i++) scanf("%lld %lld %lld %lld %lld",&lo[i],&hi[i],&d[i],&D[i],&w[i]);
    for(int i=0;i<N;i++) printf("1 0 %lld\n", hi[1]+500); // duplicate id + temp above window
    return 0;
}
