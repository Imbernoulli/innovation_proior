#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    if (!(cin >> t)) return 0;
    vector<int> ns(t);
    int maxN = 1;
    for (int i = 0; i < t; i++) {
        cin >> ns[i];
        maxN = max(maxN, ns[i]);
    }

    // Linear sieve of Euler's totient up to maxN.
    vector<int> phi(maxN + 1);
    vector<int> primes;
    vector<char> isComp(maxN + 1, 0);
    phi[1] = 1;
    for (int i = 2; i <= maxN; i++) {
        if (!isComp[i]) {
            primes.push_back(i);
            phi[i] = i - 1;
        }
        for (int p : primes) {
            long long ip = (long long)i * p;
            if (ip > maxN) break;
            isComp[ip] = 1;
            if (i % p == 0) {
                phi[ip] = phi[i] * p;
                break;
            } else {
                phi[ip] = phi[i] * (p - 1);
            }
        }
    }

    // pref[k] = sum_{q=1}^{k} phi(q)
    vector<long long> pref(maxN + 1, 0);
    for (int i = 1; i <= maxN; i++) pref[i] = pref[i - 1] + phi[i];

    // Distinct value count = 2 * (sum_{q=1}^{N} phi(q)) - 1.
    // (1/1 counted once; each coprime p<q gives p/q and q/p.)
    for (int i = 0; i < t; i++) {
        int N = ns[i];
        long long ans = 2 * pref[N] - 1;
        cout << ans << "\n";
    }
    return 0;
}
