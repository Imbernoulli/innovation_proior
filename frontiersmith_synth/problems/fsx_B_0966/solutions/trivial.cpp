// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n; ll V, LAMBDA;
    cin >> n >> V >> LAMBDA;
    ll decay, r0; cin >> decay >> r0;
    ll cheat, ccool, theat, tcool, base, ppu;
    cin >> cheat >> ccool >> theat >> tcool >> base >> ppu;
    vector<ll> sz(n + 1);
    for (int i = 1; i <= n; i++) { ll T, d, w; cin >> T >> sz[i] >> d >> w; }

    // do-nothing-clever baseline: naive sequential first-fit in raw index
    // order, ignoring both temperature and due dates.
    vector<vector<int>> batches;
    vector<int> cur; ll curSize = 0;
    for (int i = 1; i <= n; i++) {
        if (curSize + sz[i] > V && !cur.empty()) { batches.push_back(cur); cur.clear(); curSize = 0; }
        cur.push_back(i); curSize += sz[i];
    }
    if (!cur.empty()) batches.push_back(cur);

    printf("%d\n", (int)batches.size());
    for (auto &b : batches) {
        printf("%d", (int)b.size());
        for (int id : b) printf(" %d", id);
        printf("\n");
    }
    return 0;
}
