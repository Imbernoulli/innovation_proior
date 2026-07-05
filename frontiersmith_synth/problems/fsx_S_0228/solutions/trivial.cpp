// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
int main() {
    long long T, M, S, D, P, h, H;
    scanf("%lld %lld %lld %lld %lld %lld %lld", &T, &M, &S, &D, &P, &h, &H);
    for (long long i = 0; i < T; i++) {
        long long a; int av;
        scanf("%lld %d", &a, &av);
    }
    // run the on-demand gondola every slot = the scoring baseline
    for (long long i = 0; i < T; i++) printf("1 0\n");
    return 0;
}
