// TIER: trivial
// Index-order shelf packing -- exactly the checker's baseline B. Scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int W, H, N, M, R;
    scanf("%d %d %d %d %d", &W, &H, &N, &M, &R);
    vector<ll> w(N), h(N), p(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld %lld", &w[i], &h[i], &p[i]);
    for (int e = 0; e < M; e++) { int k; scanf("%d", &k); for (int j = 0; j < k; j++) { int a; scanf("%d", &a); } }
    ll cx = 0, cy = 0, rowh = 0;
    for (int i = 0; i < N; i++) {
        if (cx + w[i] > W) { cy += rowh; cx = 0; rowh = 0; }
        printf("%lld %lld\n", cx, cy);
        cx += w[i];
        rowh = max(rowh, h[i]);
    }
    return 0;
}
