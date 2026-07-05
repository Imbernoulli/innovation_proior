// TIER: trivial
// Ship the all-0 (low-gain) calibration: matches the checker's baseline B exactly -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 0; i < m; i++) {
        int w, L; scanf("%d %d", &w, &L);
        for (int j = 0; j < L; j++) { int lit; scanf("%d", &lit); }
    }
    for (int i = 0; i < n; i++) printf("%d%c", 0, i + 1 < n ? ' ' : '\n');
    return 0;
}
