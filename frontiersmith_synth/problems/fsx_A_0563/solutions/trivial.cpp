// TIER: trivial
// Monotone coordinate sweep from 0, never recalibrate.
// This reproduces the checker's internal baseline B exactly -> ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N; ll P,k; int K; ll R,Tb,SCALE,Dmax;
    scanf("%d %lld %lld %d %lld %lld %lld %lld",&N,&P,&k,&K,&R,&Tb,&SCALE,&Dmax);
    vector<ll> px(N+1), pv(N+1);
    for(int i=1;i<=N;i++) scanf("%lld %lld",&px[i],&pv[i]);
    // (accuracy table not needed for the trivial construction; ignore the rest)
    vector<int> ord(N); for(int i=0;i<N;i++) ord[i]=i+1;
    sort(ord.begin(),ord.end(),[&](int a,int b){ if(px[a]!=px[b]) return px[a]<px[b]; return a<b;});
    printf("%d\n", N);
    for(int id: ord) printf("1 %d\n", id);
    return 0;
}
