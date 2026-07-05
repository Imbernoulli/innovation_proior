// TIER: invalid
// Violates the warm-up (min-up) constraint on purpose: generator 0 (a baseload
// unit with min-up >= 4) runs for a single step then shuts off, while everything
// else stays on so capacity is never the issue. The checker rejects the too-short
// run -> score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int T, G;
    if (scanf("%d %d", &T, &G) != 2) return 0;
    for (int g = 0; g < G; g++) { ll p,b,r,k,u; scanf("%lld %lld %lld %lld %lld",&p,&b,&r,&k,&u); }
    for (int t = 0; t < T; t++) { ll d,w; scanf("%lld %lld",&d,&w); }
    for (int t = 0; t < T; t++) {
        for (int g = 0; g < G; g++) {
            if (g) fputc(' ', stdout);
            int on = (g == 0) ? (t == 0 ? 1 : 0) : 1;   // gen 0 on only at step 0
            fputc(on ? '1':'0', stdout);
        }
        fputc('\n', stdout);
    }
    return 0;
}
