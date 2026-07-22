// TIER: invalid
// Deliberately infeasible: claims a "field" with id N, which is out of the
// valid node range [0, N-1] -- the checker's bounded read rejects it
// immediately, regardless of the instance's actual topology.
#include <cstdio>
int main(){
    int N, M, F; long long D, CAP, L;
    scanf("%d %d %d %lld %lld %lld", &N, &M, &F, &D, &CAP, &L);
    printf("1\n");
    printf("%d 100.0 2 0 %d\n", N, N);
    return 0;
}
