// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// String no ropes at all: the do-nothing construction the checker itself
// measures as Bval. Must reproduce ratio ~= 0.1.
int main() {
    long long M, Q, B;
    if (!(cin >> M >> Q >> B)) return 0;
    for (long long c = 0; c < M; c++) cout << 0 << "\n";
    return 0;
}
