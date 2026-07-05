// TIER: invalid
// Deliberately infeasible: places every block far outside the die (x = W + 5) -> checker rejects.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    int W,H,N,M,R; scanf("%d %d %d %d %d",&W,&H,&N,&M,&R);
    vector<ll> w(N),h(N),p(N);
    for(int i=0;i<N;i++) scanf("%lld %lld %lld",&w[i],&h[i],&p[i]);
    for(int e=0;e<M;e++){int k;scanf("%d",&k);for(int j=0;j<k;j++){int a;scanf("%d",&a);}}
    for(int i=0;i<N;i++) printf("%lld %lld\n",(ll)W+5,(ll)H+5);
    return 0;
}
