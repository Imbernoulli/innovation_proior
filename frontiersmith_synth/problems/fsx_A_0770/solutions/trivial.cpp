// TIER: trivial
// Install nothing -- reproduces the checker's baseline B exactly -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int W,H,sx,sy; scanf("%d %d",&W,&H); scanf("%d %d",&sx,&sy);
    int NL; scanf("%d",&NL);
    for(int i=0;i<NL;i++){ int a,b; scanf("%d %d",&a,&b); }
    int NM; scanf("%d",&NM);
    for(int k=0;k<NM;k++){ ll a,b,c,d; scanf("%lld %lld %lld %lld",&a,&b,&c,&d); }
    ll Budget; scanf("%lld",&Budget);
    int M; scanf("%d",&M);
    for(int i=0;i<M;i++){
        int id,x,y; ll cost,alpha; scanf("%d %d %d %lld %lld",&id,&x,&y,&cost,&alpha);
        for(int k=0;k<NM;k++){ ll v; scanf("%lld",&v); }
    }
    printf("0\n");
    return 0;
}
