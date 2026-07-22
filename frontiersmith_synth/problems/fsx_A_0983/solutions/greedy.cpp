// TIER: greedy
// The obvious approach: at each step, among the (at most two) currently
// exposed creases -- the left frontier and the right frontier -- commit
// whichever has the BIGGER weight right now, using its required direction.
// This "grab the best available option" instinct is exactly backwards for an
// objective that multiplies weight by commit STEP (bigger weight wants a LATE
// step, not an early grab), and it never reasons about the global M/V parity
// target T at all (always folds everything with req_i).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    ll n, T;
    if (scanf("%lld %lld", &n, &T) != 2) return 0;
    ll km1 = n - 1;
    vector<string> req(km1 + 1);
    for (ll i = 1; i <= km1; i++){
        char buf[4]; scanf("%s", buf); req[i] = buf;
    }
    vector<ll> w(km1 + 1);
    for (ll i = 1; i <= km1; i++) scanf("%lld", &w[i]);

    vector<pair<ll,string>> out;
    ll L = 1, R = km1;
    while (L <= R){
        if (L == R){ out.push_back({L, req[L]}); break; }
        if (w[L] >= w[R]){ out.push_back({L, req[L]}); L++; }
        else { out.push_back({R, req[R]}); R--; }
    }

    printf("%zu\n", out.size());
    for (auto &pr : out) printf("%lld %s\n", pr.first, pr.second.c_str());
    return 0;
}
