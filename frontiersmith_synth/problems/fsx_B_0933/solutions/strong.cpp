// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Insight: for a monomial map x -> x^d, the difference pi(x+s)-pi(x) for a fixed
// nonzero shift s is (x+s)^d - x^d. Substituting u = x/s (s invertible mod prime
// p) shows this equals s^d * ((u+1)^d - u^d): the SHAPE of the difference
// distribution is identical for every shift s (only rescaled). So the peak
// leakage of x^d is computable in O(p) from a SINGLE canonical polynomial
// g(u) = (u+1)^d - u^d, instead of the O(p^2) brute-force DDT a non-algebraic
// search would need per candidate. That lets us SCAN many candidate exponents d
// (each must satisfy gcd(d,p-1)=1 to be a bijection) and keep the one with the
// smallest peak leakage -- not just the smallest valid d (which is often but not
// always optimal: when p-1 has several small prime factors, small exponents are
// unavailable and the best remaining exponent is not obvious without scanning).
//
// x^d always fixes 0 and 1 (0^d=0, 1^d=1), so raw x^d is never a derangement.
// Multiplicative/additive affine dressing pi(x) = a*x^d + c does not change the
// peak leakage at all (a permutes the difference buckets, c only shifts labels),
// so we search (a,c) purely for feasibility: for a fixed a, every x contributes
// exactly one FORBIDDEN c value (the one that would fix x); once those are
// marked, any free c works, found in O(p).

static ll powmod(ll b, ll e, ll m){
    b %= m; if (b < 0) b += m;
    ll r = 1;
    while (e > 0){ if (e & 1) r = r * b % m; b = b * b % m; e >>= 1; }
    return r;
}

int main(){
    ll p;
    scanf("%lld", &p);

    int scanMax = (int)min((ll)1000, p - 2);

    int bestD = -1, bestU = INT_MAX;
    vector<int> cnt((size_t)p, 0);
    for (int d = 3; d <= scanMax; d += 2){
        if (__gcd((ll)d, p - 1) != 1) continue;
        fill(cnt.begin(), cnt.end(), 0);
        int u = 0;
        for (ll x = 0; x < p; x++){
            ll g = (powmod(x + 1, d, p) - powmod(x, d, p)) % p;
            if (g < 0) g += p;
            int c = ++cnt[(size_t)g];
            if (c > u) u = c;
        }
        if (u < bestU){ bestU = u; bestD = d; }
    }
    if (bestD < 0){
        // extremely unlikely fallback: no small odd exponent is coprime to p-1
        // within the scan window -- widen the search to the full range.
        for (int d = 3; d < p - 1; d += 2){
            if (__gcd((ll)d, p - 1) != 1) continue;
            fill(cnt.begin(), cnt.end(), 0);
            int u = 0;
            for (ll x = 0; x < p; x++){
                ll g = (powmod(x + 1, d, p) - powmod(x, d, p)) % p;
                if (g < 0) g += p;
                int c = ++cnt[(size_t)g];
                if (c > u) u = c;
            }
            if (u < bestU){ bestU = u; bestD = d; }
        }
    }

    int d = bestD;
    ll chosenA = -1, chosenC = -1;
    vector<char> forbidden((size_t)p, 0);
    for (ll a = 1; a <= 50 && chosenA < 0; a++){
        fill(forbidden.begin(), forbidden.end(), 0);
        for (ll x = 0; x < p; x++){
            ll xr = powmod(x, d, p);
            ll h = (a * xr - x) % p; if (h < 0) h += p;
            ll badc = (p - h) % p;         // c that would make x a fixed point
            forbidden[(size_t)badc] = 1;
        }
        for (ll c = 0; c < p; c++){
            if (!forbidden[(size_t)c]){ chosenA = a; chosenC = c; break; }
        }
    }
    // defensive fallback (should not trigger in practice)
    if (chosenA < 0){ chosenA = 1; chosenC = 0; }

    vector<ll> pi((size_t)p);
    for (ll x = 0; x < p; x++)
        pi[(size_t)x] = (chosenA * powmod(x, d, p) + chosenC) % p;

    for (ll x = 0; x < p; x++) printf("%lld%c", pi[(size_t)x], x + 1 < p ? ' ' : '\n');
    return 0;
}
