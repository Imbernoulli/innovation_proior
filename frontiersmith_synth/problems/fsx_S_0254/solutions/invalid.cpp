// TIER: invalid
// Deliberately infeasible: claims one sensor but prints an out-of-range pool index.
// The checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M; long long R;
    if (scanf("%d %d %lld", &N, &M, &R) != 3) return 0;
    printf("1\n%d\n", N + 1); // index N+1 is out of range [1,N]
    return 0;
}
