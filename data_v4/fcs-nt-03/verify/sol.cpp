#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n;
    if (!(cin >> n)) return 0;          // no input -> nothing to do
    if (n <= 0) { cout << 0 << "\n"; return 0; } // D(n)=0 for n<=0 by convention

    // Dirichlet hyperbola method for D(n) = sum_{i=1..n} d(i) = sum_{i=1..n} floor(n/i).
    // Pairs (a,b) with a*b <= n are counted by symmetry across the line a=b:
    //   D(n) = 2 * sum_{i=1..s} floor(n/i) - s*s,  where s = floor(sqrt(n)).
    long long s = (long long)sqrtl((long double)n);
    while (s * s > n) s--;               // guard sqrt rounding from above
    while ((s + 1) * (s + 1) <= n) s++;  // guard sqrt rounding from below

    long long sum = 0;
    for (long long i = 1; i <= s; i++) sum += n / i;

    long long answer = 2 * sum - s * s;
    cout << answer << "\n";
    return 0;
}
