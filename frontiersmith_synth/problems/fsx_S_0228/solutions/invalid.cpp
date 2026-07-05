// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
int main() {
    long long T, M, S, D, P, h, H;
    scanf("%lld %lld %lld %lld %lld %lld %lld", &T, &M, &S, &D, &P, &h, &H);
    for (long long i = 0; i < T; i++) {
        long long a; int av;
        scanf("%lld %d", &a, &av);
    }
    // deliberately out of range: on = 2 (not in {0,1}) -> checker rejects, score 0
    printf("2 0\n");
    for (long long i = 1; i < T; i++) printf("1 0\n");
    return 0;
}
