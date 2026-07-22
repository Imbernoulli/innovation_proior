// TIER: trivial
// Replays the given chord verbatim: no cancellation attempted at all. This is
// exactly the checker's internal baseline construction, so it always scores
// ratio == 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<ll> a(n);
    ll S = 0;
    for (int i = 0; i < n; i++) { cin >> a[i]; S += a[i]; }

    cout << S << "\n";
    for (int i = 0; i < n; i++)
        for (ll k = 0; k < a[i]; k++) cout << i << ' ';
    cout << "\n";
    return 0;
}
