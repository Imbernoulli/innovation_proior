// TIER: invalid
// Deliberately infeasible: prints an out-of-range token (2, not in {0,1}) as the very
// first wall bit. The checker's bounded read rejects it immediately -> no Ratio -> 0.
#include <cstdio>
#include <iostream>
using namespace std;
int main() {
    int n; long long a, b, c;
    cin >> n >> a >> b >> c;
    // first H row: one out-of-range bit, rest arbitrary
    cout << 2;
    for (int c2 = 1; c2 < n - 1; c2++) cout << ' ' << 0;
    cout << '\n';
    for (int r = 1; r < n; r++) {
        for (int c2 = 0; c2 < n - 1; c2++) cout << (c2 ? " " : "") << 0;
        cout << '\n';
    }
    for (int r = 0; r + 1 < n; r++) {
        for (int c2 = 0; c2 < n; c2++) cout << (c2 ? " " : "") << 0;
        cout << '\n';
    }
    return 0;
}
