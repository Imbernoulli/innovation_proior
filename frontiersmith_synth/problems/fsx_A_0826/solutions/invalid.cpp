// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: prints a digester coordinate out of the allowed [0,100000]
// range, which the checker must reject regardless of M -> must score 0.
int main() {
    int M; long long A, Bc, Lm;
    if (!(cin >> M >> A >> Bc >> Lm)) { cout << "1\n-5 -5\n1\n"; return 0; }
    for (int i = 0; i < M; i++) {
        long long x, y, w; cin >> x >> y >> w;
    }
    cout << 1 << "\n";
    cout << -5 << " " << -5 << "\n";   // out-of-range coordinate -> infeasible
    for (int i = 0; i < M; i++) cout << 1 << (i + 1 < M ? ' ' : '\n');
    return 0;
}
