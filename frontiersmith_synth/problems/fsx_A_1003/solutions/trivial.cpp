// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Grabs EVERY guaranteed zero-slew, zero-decay freebie pair (the same
// construction the checker sums for its internal baseline B) and leaves
// every other telescope idle. F == B exactly -> Ratio == 0.1.

int main() {
    int T, M, H, W; ll Pn, Qd;
    cin >> T >> M >> H >> W >> Pn >> Qd;
    vector<int> site(T), pos(T), speed(T);
    for (int i = 0; i < T; i++) cin >> site[i] >> pos[i] >> speed[i];
    vector<int> a(M), pt(M), v(M), o(M);
    for (int j = 0; j < M; j++) cin >> a[j] >> pt[j] >> v[j] >> o[j];

    // one dedicated telescope pair per qualifying freebie target
    vector<int> use0(M, -1), use1(M, -1);
    vector<char> isFreebie(M, 0);
    vector<char> claimed(T, 0);
    for (int j = 0; j < M; j++) {
        if (a[j] != 0) continue;
        int i0 = -1, i1 = -1;
        for (int i = 0; i < T; i++) {
            if (claimed[i] || pos[i] != pt[j]) continue;
            if (site[i] == 0 && i0 < 0) i0 = i;
            if (site[i] == 1 && i1 < 0) i1 = i;
        }
        if (i0 >= 0 && i1 >= 0) {
            isFreebie[j] = 1; use0[j] = i0; use1[j] = i1;
            claimed[i0] = claimed[i1] = 1;
        }
    }

    vector<int> myTarget(T, -1);
    for (int j = 0; j < M; j++) if (isFreebie[j]) { myTarget[use0[j]] = j; myTarget[use1[j]] = j; }

    for (int i = 0; i < T; i++) {
        if (myTarget[i] >= 0) cout << 1 << "\n" << 0 << " " << myTarget[i] << "\n";
        else cout << 0 << "\n";
    }
    return 0;
}
