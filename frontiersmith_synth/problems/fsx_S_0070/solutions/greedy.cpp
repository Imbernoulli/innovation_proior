// TIER: greedy
// One-pass coordinate greedy from the all-Side baseline: visit acts in order of
// descending "Main-Stage appeal" (total weight of clauses that want them on Main),
// and move each act to Main iff that strictly increases satisfied weight given the
// choices made so far. A single sweep -- no re-visiting.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> cw;
vector<vector<int>> clit;
vector<vector<int>> occ;   // var -> clause ids
vector<int> a;             // assignment 1..n
vector<int> satc;          // satisfied-literal count per clause

// gain of flipping variable v (0<->1) under current assignment
ll gainFlip(int v) {
    ll delta = 0;
    int cur = a[v];
    for (int c : occ[v]) {
        // find v's literal sign in clause c
        for (int L : clit[c]) {
            if (abs(L) == v) {
                bool truthNow = (L > 0) ? (cur == 1) : (cur == 0);
                if (truthNow) {
                    if (satc[c] == 1) delta -= cw[c]; // this literal is the only support
                } else {
                    if (satc[c] == 0) delta += cw[c]; // clause currently unsatisfied
                }
                break;
            }
        }
    }
    return delta;
}

void applyFlip(int v) {
    int cur = a[v];
    for (int c : occ[v]) {
        for (int L : clit[c]) {
            if (abs(L) == v) {
                bool truthNow = (L > 0) ? (cur == 1) : (cur == 0);
                if (truthNow) satc[c] -= 1; else satc[c] += 1;
                break;
            }
        }
    }
    a[v] = 1 - cur;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    cw.resize(m); clit.resize(m); occ.assign(n + 1, {});
    for (int c = 0; c < m; c++) {
        int k, w; scanf("%d %d", &k, &w); cw[c] = w; clit[c].resize(k);
        for (int t = 0; t < k; t++) { int L; scanf("%d", &L); clit[c][t] = L; occ[abs(L)].push_back(c); }
    }
    a.assign(n + 1, 0);
    satc.assign(m, 0);
    // init satc for all-zero
    for (int c = 0; c < m; c++) {
        int s = 0;
        for (int L : clit[c]) if (L < 0) s++;   // negative literal true when var=0
        satc[c] = s;
    }
    // appeal = total weight of clauses where +v appears
    vector<ll> appeal(n + 1, 0);
    for (int c = 0; c < m; c++)
        for (int L : clit[c]) if (L > 0) appeal[L] += cw[c];
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int x, int y){ return appeal[x] > appeal[y]; });

    for (int v : order) {
        if (gainFlip(v) > 0) applyFlip(v);
    }

    for (int i = 1; i <= n; i++) printf("%d%c", a[i], i == n ? '\n' : ' ');
    return 0;
}
