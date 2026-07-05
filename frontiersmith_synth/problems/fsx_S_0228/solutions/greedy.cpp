// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
int main() {
    long long T, M, S, D, P, h, H;
    scanf("%lld %lld %lld %lld %lld %lld %lld", &T, &M, &S, &D, &P, &h, &H);
    vector<long long> a(T);
    vector<int> av(T);
    for (long long i = 0; i < T; i++) scanf("%lld %d", &a[i], &av[i]);
    // clear every slot immediately with the cheapest single lift that fits
    for (long long i = 0; i < T; i++) {
        long long demand = a[i]; // greedy keeps queue empty, so demand is this slot's arrivals
        if (demand == 0) { printf("0 0\n"); continue; }
        if (av[i] && S >= demand) printf("0 1\n");   // spot alone clears it, and P < D
        else printf("1 0\n");                          // on-demand (M >= a always)
    }
    return 0;
}
