#include <bits/stdc++.h>
using namespace std;

static long long gcdll(long long a, long long b) {
    while (b) { long long t = a % b; a = b; b = t; }
    return a;
}

int main() {
    long long n;
    if (!(cin >> n)) return 0;

    // Count unordered pairs {x, y}, 1 <= x < y <= n, with lcm(x, y) <= n.
    // Write x = g*a, y = g*b, g = gcd(x, y), gcd(a, b) = 1. Then x < y  <=>  a < b,
    // and lcm = g*a*b. The constraint lcm = g*a*b <= n already forces both x, y <= n
    // (since lcm >= y > x). So the count is:
    //     sum over a < b with gcd(a, b) = 1 and a*b <= n  of  floor(n / (a*b)).
    // The a < b ordering is what makes each unordered pair counted exactly once.
    long long ans = 0;
    for (long long a = 1; a * (a + 1) <= n; a++) {       // need a < b  =>  a*b >= a*(a+1)
        for (long long b = a + 1; a * b <= n; b++) {     // strictly a < b: no a == b term
            if (gcdll(a, b) != 1) continue;              // a, b must be coprime
            ans += n / (a * b);                          // g ranges 1..floor(n/(a*b))
        }
    }

    cout << ans << "\n";
    return 0;
}
