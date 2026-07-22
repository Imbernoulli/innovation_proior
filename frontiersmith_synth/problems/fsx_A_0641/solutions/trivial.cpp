// TIER: trivial
// Do-nothing shuffle: output the identity permutation. Every channel's correlation is at
// its provable maximum (|T(d)|=p for all d simultaneously), which is exactly the checker's
// own baseline B -- so this scores the calibration point (ratio = 0.1) by construction.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int p;
    if (scanf("%d", &p) != 1) return 0;
    for (int d = 1; d <= p - 1; d++) { int w; scanf("%d", &w); }
    for (int i = 0; i < p; i++) printf("%d%c", i, i + 1 == p ? '\n' : ' ');
    return 0;
}
