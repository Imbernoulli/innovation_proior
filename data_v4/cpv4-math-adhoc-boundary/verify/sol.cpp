#include <bits/stdc++.h>
using namespace std;

// Count x in [1, n] with floor(n/x) in [a, b], a<=b.
// floor(n/x) >= a  <=>  x <= floor(n/a)            (for a >= 1; if a <= 0 every x qualifies)
// floor(n/x) <= b  <=>  x >  floor(n/(b+1))        (since floor(n/x) <= b iff x > n/(b+1))
// So x ranges over ( floor(n/(b+1)) , floor(n/a) ], a left-open / right-closed interval.
long long solve(long long n, long long a, long long b) {
    if (n <= 0) return 0;
    if (a > b) return 0;
    if (b < 0) return 0;            // floor(n/x) >= 0 always; cannot be <= negative b
    // Upper index bound from the a-condition (floor(n/x) >= a).
    long long hi;
    if (a <= 0) {
        hi = n;                     // floor(n/x) >= a holds for all x in [1, n]
    } else {
        hi = n / a;                 // largest x with floor(n/x) >= a
    }
    if (hi > n) hi = n;
    // Lower exclusive bound from the b-condition (floor(n/x) <= b).
    long long loExcl = n / (b + 1); // floor(n/x) <= b iff x > floor(n/(b+1))
    if (loExcl < 0) loExcl = 0;
    // x in (loExcl, hi], also forced into [1, n] by construction.
    long long cnt = hi - loExcl;
    if (cnt < 0) cnt = 0;
    return cnt;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        long long n, a, b;
        cin >> n >> a >> b;
        cout << solve(n, a, b) << "\n";
    }
    return 0;
}
