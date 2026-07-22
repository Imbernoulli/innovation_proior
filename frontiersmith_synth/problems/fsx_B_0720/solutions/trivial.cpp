// TIER: trivial
// Earliest-start for everyone: s_i = e_i. This is exactly the checker's own
// baseline construction (no reasoning about temperature, value, or order).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N; ll H, C;
    scanf("%d %lld %lld", &N, &H, &C);
    vector<ll> e(N), l(N);
    for (int i = 0; i < N; i++){
        int type; ll vol, p1, p2;
        scanf("%d %lld %lld %lld %lld %lld", &type, &vol, &e[i], &l[i], &p1, &p2);
    }
    for (int i = 0; i < N; i++) printf("%lld\n", e[i]);
    return 0;
}
