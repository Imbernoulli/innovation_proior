// TIER: trivial
// Do nothing: drop zero grains everywhere. The stabilized grid stays empty, which
// matches exactly the zero cells of the target -> reproduces the checker baseline B.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N,K; long long B;
    if(scanf("%d %d %lld",&N,&K,&B)!=3) return 0;
    for(int i=0;i<K;i++) printf("%d%c",0,i+1<K?' ':'\n');
    return 0;
}
