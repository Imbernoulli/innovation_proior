// TIER: invalid
// Deliberately infeasible: emits a loop referencing an out-of-range rig index -> scores 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    ll N,M,C,p;
    if(scanf("%lld %lld %lld %lld",&N,&M,&C,&p)!=4) return 0;
    for(ll i=1;i<=N;i++){ ll a,b,cc; scanf("%lld %lld %lld",&a,&b,&cc); }
    // one loop that names a nonexistent rig (index N+... encoded as a huge number)
    printf("1\n");
    printf("2 1 1000000000\n");   // rig 1000000000 does not exist (and rig 1 reused) -> infeasible
    return 0;
}
