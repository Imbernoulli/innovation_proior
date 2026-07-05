// TIER: trivial
// Reproduces the judge's internal baseline exactly: a regular bounding-box grid
// tiling of array type 1 (orient 0), row-major, capped by availability.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int H, W, T;
    if (scanf("%d %d %d", &H, &W, &T) != 3) return 0;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) { int x; scanf("%d", &x); }
    int avail1 = 0, bh = 0, bw = 0;
    for (int t = 1; t <= T; t++) {
        int k, c; scanf("%d %d", &k, &c);
        for (int i = 0; i < k; i++) {
            int rr, cc; scanf("%d %d", &rr, &cc);
            if (t == 1) { bh = max(bh, rr); bw = max(bw, cc); }
        }
        if (t == 1) avail1 = c;
    }
    bh++; bw++;
    ll rows = H / bh, cols = W / bw;
    ll slots = rows * cols;
    ll copies = min((ll)avail1, slots);

    vector<array<ll,4>> out;
    ll placed = 0;
    for (ll i = 0; i < rows && placed < copies; i++)
        for (ll j = 0; j < cols && placed < copies; j++) {
            out.push_back({1, 0, i * bh, j * bw});
            placed++;
        }
    printf("%d\n", (int)out.size());
    for (auto& p : out) printf("%lld %lld %lld %lld\n", p[0], p[1], p[2], p[3]);
    return 0;
}
