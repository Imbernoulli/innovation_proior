#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef __int128 lll;

int n;
vector<ll> e1, g1, e2, g2;   // +1 hop from i (i=0..n-2); +2 hop from i (i=0..n-3)

// Inner shortest-path DP, parameter lambda = P/Q (Q > 0).
// Edge weight w = e*Q - g*P. Find the path 0 -> n-1 of MINIMUM total weight,
// and return the (E = sum e, D = sum g) realizing it.
struct St { lll w; lll E; lll D; };

void inner(lll P, lll Q, lll &outE, lll &outD) {
    // dp[i] = best (min-weight) state to reach stone i. Reachable stones only.
    vector<St> dp(n);
    vector<char> seen(n, 0);
    dp[0] = St{0, 0, 0};
    seen[0] = 1;
    for (int i = 0; i < n; i++) {
        if (!seen[i]) continue;
        // +1 hop  i -> i+1
        if (i + 1 <= n - 1) {
            lll w = (lll)e1[i] * Q - (lll)g1[i] * P;   // can OVERFLOW ll
            St cand{ dp[i].w + w, dp[i].E + e1[i], dp[i].D + g1[i] };
            if (!seen[i+1] || cand.w < dp[i+1].w) { dp[i+1] = cand; seen[i+1] = 1; }
        }
        // +2 hop  i -> i+2
        if (i + 2 <= n - 1) {
            lll w = (lll)e2[i] * Q - (lll)g2[i] * P;
            St cand{ dp[i].w + w, dp[i].E + e2[i], dp[i].D + g2[i] };
            if (!seen[i+2] || cand.w < dp[i+2].w) { dp[i+2] = cand; seen[i+2] = 1; }
        }
    }
    outE = dp[n-1].E; outD = dp[n-1].D;
}

int main() {
    if (scanf("%d", &n) != 1) return 0;
    e1.assign(max(0, n-1), 0); g1.assign(max(0, n-1), 0);
    e2.assign(max(0, n-2), 0); g2.assign(max(0, n-2), 0);
    for (int i = 0; i < n-1; i++) { long long a,b; if(scanf("%lld %lld",&a,&b)!=2) return 0; e1[i]=a; g1[i]=b; }
    for (int i = 0; i < n-2; i++) { long long a,b; if(scanf("%lld %lld",&a,&b)!=2) return 0; e2[i]=a; g2[i]=b; }

    // Dinkelbach for MINIMIZATION. Start lambda from one feasible path's ratio:
    // the all-(+1) path 0->1->...->(n-1).
    lll P = 0, Q = 0;
    for (int i = 0; i < n-1; i++) { P += e1[i]; Q += g1[i]; }
    // (n>=2 guaranteed, so this path exists and Q>0)

    while (true) {
        lll E, D;
        inner(P, Q, E, D);
        // g(lambda) = E*Q - D*P  (<=0 always at/after start; ==0 means optimal)
        lll g = E * Q - D * P;
        if (g >= 0) break;     // no strictly cheaper path => P/Q is the minimum pace
        P = E; Q = D;          // next lambda = E/D (strictly smaller)
    }

    // reduce P/Q
    auto absll = [](lll x){ return x < 0 ? -x : x; };
    lll a = absll(P), b = absll(Q);
    while (b) { lll t = a % b; a = b; b = t; }
    lll g = a ? a : 1;
    P /= g; Q /= g;

    auto out128 = [](lll x){
        if (x == 0) { putchar('0'); return; }
        if (x < 0) { putchar('-'); x = -x; }
        char buf[48]; int len = 0;
        while (x > 0) { buf[len++] = char('0' + (int)(x % 10)); x /= 10; }
        while (len) putchar(buf[--len]);
    };
    out128(P); putchar(' '); out128(Q); putchar('\n');
    return 0;
}
