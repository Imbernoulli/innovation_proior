// TIER: greedy
// Weighted majority voting: set each sensor to the mode whose incident literals carry the
// greater total requirement weight. One independent pass; ignores clause-level interactions.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<ll> wpos(n + 1, 0), wneg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int w, L; scanf("%d %d", &w, &L);
        for (int j = 0; j < L; j++) {
            int lit; scanf("%d", &lit);
            int v = abs(lit);
            if (lit > 0) wpos[v] += w; else wneg[v] += w;
        }
    }
    for (int i = 1; i <= n; i++) {
        int a = (wpos[i] > wneg[i]) ? 1 : 0;
        printf("%d%c", a, i < n ? ' ' : '\n');
    }
    return 0;
}
