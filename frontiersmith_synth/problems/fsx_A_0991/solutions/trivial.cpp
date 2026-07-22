// TIER: trivial
// Do-nothing baseline: every component on one single card, no braids.
// This reproduces the checker's internal baseline B exactly -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int F, Q, R; long long P, K;
    if (scanf("%d %d %lld %lld %d", &F, &Q, &P, &K, &R) != 5) return 0;
    for (int i = 0; i < F; i++){ long long w; scanf("%lld", &w); }
    for (int i = 1; i <= F; i++) printf("1 0\n");
    return 0;
}
