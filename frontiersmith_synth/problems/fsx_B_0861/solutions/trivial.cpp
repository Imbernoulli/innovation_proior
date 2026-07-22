// TIER: trivial
// The baseline reference itself: visit the stops in raw index order 2,3,...,N.
// This reproduces exactly the checker's internal baseline B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N, M, K;
    scanf("%d %d %d", &N, &M, &K);
    for (int i = 0; i < M; i++){ long long u,v,w; scanf("%lld %lld %lld", &u,&v,&w); }
    for (int s = 0; s < K; s++){
        int B; long long w; scanf("%d %lld", &B, &w);
        for (int j = 0; j < B; j++){ int id; scanf("%d", &id); }
    }
    for (int i = 2; i <= N; i++) printf("%d%c", i, i<N?' ':'\n');
    return 0;
}
