// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: seats every metronome at seat 1 (not a permutation).
int main() {
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++) cout << 1 << (i < n ? ' ' : '\n');
    return 0;
}
