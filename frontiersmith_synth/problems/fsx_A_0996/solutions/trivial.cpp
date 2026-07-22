// TIER: trivial
// Reproduces the checker's own internal baseline construction exactly: a
// short-period (period <= 6) letter-cycling pattern matching the requested
// frequencies via the exact fair-share ("deficit") rule with denominator 6.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static string deficitRational(ll L, int a, const vector<ll>& num, ll den) {
    vector<ll> cnt(a, 0);
    string w(L, '0');
    for (ll i = 0; i < L; i++) {
        int best = 0;
        ll bestScore = num[0] * (i + 1) - cnt[0] * den;
        for (int c = 1; c < a; c++) {
            ll sc = num[c] * (i + 1) - cnt[c] * den;
            if (sc > bestScore) { bestScore = sc; best = c; }
        }
        cnt[best]++;
        w[i] = (char)('0' + best);
    }
    return w;
}

int main() {
    int a, K, tol_w, tol_g, w_pal;
    ll L;
    cin >> a >> L >> K >> tol_w >> tol_g >> w_pal;
    vector<ll> freq(a);
    for (int c = 0; c < a; c++) cin >> freq[c];

    const ll P0 = 6;
    vector<ll> num(a, 0);
    {
        vector<pair<double,int>> frac(a);
        ll s = 0;
        for (int c = 0; c < a; c++) {
            double target = (double)freq[c] * (double)P0 / (double)L;
            num[c] = (ll)floor(target);
            frac[c] = {target - floor(target), c};
            s += num[c];
        }
        sort(frac.rbegin(), frac.rend());
        ll rem = P0 - s;
        for (int k = 0; k < (int)frac.size() && rem > 0; k++) { num[frac[k].second]++; rem--; }
    }
    string w = deficitRational(L, a, num, P0);
    cout << w << "\n";
    return 0;
}
