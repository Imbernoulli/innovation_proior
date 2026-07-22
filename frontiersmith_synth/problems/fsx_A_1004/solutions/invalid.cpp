// TIER: invalid
// Deliberately infeasible: uses the always-structurally-legal midpoint slot
// (so it isn't rejected for a trivial parsing reason) but labels every
// crease 'M', which blatantly violates Maekawa's |#M-#V|=2 for any d>2.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int k, S, mmax, beta1000, gamma1000;
    cin >> k >> S >> mmax >> beta1000 >> gamma1000;
    vector<int> v(S, 0);
    for (int i = 1; i <= S - 1; i++) cin >> v[i];

    int mid = S / 2;
    long long d = (long long)k * 2;

    cout << 1 << "\n" << mid << "\n";
    for (long long i = 0; i < d; i++) cout << 'M' << " \n"[i + 1 == d];
    return 0;
}
