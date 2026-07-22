// TIER: invalid
// Deliberately infeasible: report a negative grain count (out of [0,B]).
// The checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N,K; long long B;
    if(scanf("%d %d %lld",&N,&K,&B)!=3) return 0;
    printf("-1");
    for(int i=1;i<K;i++) printf(" 0");
    printf("\n");
    return 0;
}
