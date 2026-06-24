#include <bits/stdc++.h>
using namespace std;

// DSU over stars 1..n, with a "next free" skip pointer to union a whole
// contiguous range in near-linear amortized total time.
struct DSU {
    vector<int> par;     // connectivity parent
    vector<int> nxt;     // nxt[i] = smallest index >= i not yet "consumed" by a range-union step
    DSU(int n) : par(n + 2), nxt(n + 2) {
        for (int i = 0; i <= n + 1; i++) { par[i] = i; nxt[i] = i; }
    }
    int find(int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a != b) par[a] = b;
    }
    // skip-pointer find over nxt[]
    int fnext(int x) { while (nxt[x] != x) { nxt[x] = nxt[nxt[x]]; x = nxt[x]; } return x; }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    DSU d(n);

    for (int op = 0; op < m; op++) {
        int l, r;
        cin >> l >> r;
        // Union all stars in [l, r] together. We walk from l upward using the
        // skip pointer nxt[] so each star is "consumed" (its nxt advanced) at
        // most once across the whole run, giving near-linear total work.
        //
        // Boundary: a range [l, r] has (r - l) adjacent links l-l+1, ..., (r-1)-r.
        // We start at the first not-yet-consumed index >= l, and keep linking it
        // to its successor while we are still strictly below r. The cursor i must
        // never pass r, and the LAST link we add is (r-1) -> r, so the loop runs
        // while i < r (NOT i <= r): linking i to i+1 when i == r would touch r+1,
        // which is outside the range.
        int i = d.fnext(l);
        while (i < r) {
            d.unite(i, i + 1);   // connect star i with star i+1
            d.nxt[i] = i + 1;    // i is consumed; future range-walks skip past it
            i = d.fnext(i + 1);  // jump to next not-yet-consumed index
        }
    }

    // Count connected components among all stars 1..n.
    int comps = 0;
    for (int v = 1; v <= n; v++)
        if (d.find(v) == v) comps++;

    cout << comps << "\n";
    return 0;
}
