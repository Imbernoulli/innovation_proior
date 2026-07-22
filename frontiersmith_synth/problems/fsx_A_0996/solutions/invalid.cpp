// TIER: invalid
// Deliberately infeasible: emits a word one character too long, so the
// checker's length check rejects it on every test (Ratio must be 0.0).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int a, K, tol_w, tol_g, w_pal;
    ll L;
    cin >> a >> L >> K >> tol_w >> tol_g >> w_pal;
    vector<ll> freq(a);
    for (int c = 0; c < a; c++) cin >> freq[c];
    string w(L + 1, '0');
    cout << w << "\n";
    return 0;
}
