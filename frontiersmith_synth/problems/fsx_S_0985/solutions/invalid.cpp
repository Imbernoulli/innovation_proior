// TIER: invalid
// Deliberately infeasible: claims one strike on a bell index that is out of
// range (e = n, but valid bells are 0..n-1). The checker's bounded read must
// reject this immediately -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    cout << 1 << "\n";
    cout << n << "\n"; // out of range: valid range is [0, n-1]
    return 0;
}
