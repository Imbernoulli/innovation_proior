// TIER: trivial
// Always claims PASS ("no winning move exists"). This is exactly the grader's own internal
// baseline construction, so it matches B on every test by definition (ratio ~= 0.1).
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long M;
    if (!(cin >> M)) return 0;
    for (long long m = 0; m < M; m++) {
        int n, c, e;
        cin >> n >> c >> e;
        for (int i = 0; i < c; i++) { int x; cin >> x; }
        for (int i = 0; i < e; i++) { int u, v; cin >> u >> v; }
        cout << "PASS\n";
    }
    return 0;
}
