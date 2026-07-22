// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, C, K;
    scanf("%d %d %d", &N, &C, &K);
    ll alpha; double theta; ll N0;
    scanf("%lld %lf %lld", &alpha, &theta, &N0);
    vector<ll> P(K + 1);
    for (int j = 1; j <= K; j++) scanf("%lld", &P[j]);
    vector<ll> tx(N + 1), ty(N + 1), rx(N + 1), ry(N + 1), w(N + 1);
    for (int i = 1; i <= N; i++)
        scanf("%lld %lld %lld %lld %lld", &tx[i], &ty[i], &rx[i], &ry[i], &w[i]);

    // Reproduce the checker's baseline: top min(N,C) values (ties -> smaller
    // index), each alone on its own channel at the top power level.
    vector<int> idx(N);
    for (int i = 0; i < N; i++) idx[i] = i + 1;
    stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return w[a] > w[b]; });
    int take = min(N, C);

    vector<int> outc(N + 1, -1), outk(N + 1, -1);
    for (int t = 0; t < take; t++) {
        int link = idx[t];
        outc[link] = t + 1;
        outk[link] = K;
    }
    for (int i = 1; i <= N; i++) printf("%d %d\n", outc[i], outk[i]);
    return 0;
}
