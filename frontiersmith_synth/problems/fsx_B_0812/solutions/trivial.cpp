// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, K; ll T;
    scanf("%d %d %lld", &N, &K, &T);
    vector<int> lens(K);
    for (int p = 0; p < K; p++) scanf("%d", &lens[p]);

    vector<int> Kv(N + 1);
    vector<array<pair<ll,ll>,3>> menu(N + 1);
    for (int v = 1; v <= N; v++) {
        int kv; scanf("%d", &kv); Kv[v] = kv;
        for (int i = 0; i < kv; i++) {
            ll d, a; scanf("%lld %lld", &d, &a);
            menu[v][i] = {d, a};
        }
    }

    vector<vector<ll>> w(K);
    for (int p = 0; p < K; p++) {
        int L = lens[p];
        w[p].assign(L - 1, 0);
        for (int i = 0; i < L - 1; i++) {
            ll c; scanf("%lld %lld", &w[p][i], &c);
        }
    }

    // x = fastest (priciest) variant everywhere
    for (int v = 1; v <= N; v++) printf("%d ", Kv[v] - 1);
    printf("\n");
    // booth counts unchanged
    for (int p = 0; p < K; p++)
        for (int i = 0; i < (int)w[p].size(); i++) printf("%lld ", w[p][i]);
    printf("\n");
    return 0;
}
