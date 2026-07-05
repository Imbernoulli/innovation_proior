// TIER: trivial
// First-fit-by-index (value-blind): reproduces exactly the checker's baseline B, ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int H, R;
    scanf("%d %d", &H, &R);
    vector<ll> rem(R + 1);
    for (int j = 1; j <= R; j++) scanf("%lld", &rem[j]);
    vector<int> ans(H + 1, 0);
    for (int i = 1; i <= H; i++) {
        int chosen = 0;
        for (int j = 1; j <= R; j++) {
            int v, d; scanf("%d %d", &v, &d);
            if (!chosen && (ll)d <= rem[j]) { chosen = j; rem[j] -= d; }
        }
        ans[i] = chosen;
    }
    for (int i = 1; i <= H; i++) printf("%d\n", ans[i]);
    return 0;
}
