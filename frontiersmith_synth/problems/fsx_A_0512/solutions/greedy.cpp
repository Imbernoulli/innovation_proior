// TIER: greedy
// The obvious approach: buy capacity proportional to each line's base load.
// Covers every base load but leaves only a thin, uniform surge cushion -> the
// backbone entrances trip in the worst cascade. It also wastes a big slice on the
// high-base-load pocket, which trips and grounds anyway.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    ll L, M, S, B;
    if (scanf("%lld %lld %lld %lld", &L, &M, &S, &B) != 4) return 0;
    vector<ll> w(L), v(L);
    ll sumW = 0;
    for (ll i = 0; i < L; i++){ scanf("%lld %lld", &w[i], &v[i]); sumW += w[i]; }
    // (edges + scenarios not needed by this heuristic)
    vector<ll> cap(L, 0);
    ll used = 0;
    if (sumW <= 0){
        ll U = B / L;
        for (ll i = 0; i < L; i++) cap[i] = U;
        used = U * L;
    } else {
        for (ll i = 0; i < L; i++){
            cap[i] = (ll)((__int128)B * w[i] / sumW);
            used += cap[i];
        }
    }
    // hand leftover budget to the highest-load line (still proportional in spirit)
    ll left = B - used;
    if (left > 0){
        ll bi = 0; for (ll i = 1; i < L; i++) if (w[i] > w[bi]) bi = i;
        cap[bi] += left;
    }
    for (ll i = 0; i < L; i++) printf("%lld%c", cap[i], i + 1 < L ? ' ' : '\n');
    return 0;
}
