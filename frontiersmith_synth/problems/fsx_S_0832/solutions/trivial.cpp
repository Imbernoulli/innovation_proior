// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
// Identity coupling: P-node i <-> C-node i. This is exactly the checker's own
// baseline construction, so it must score ratio ~0.1 on every test.
int main() {
    int N;
    if (!(cin >> N)) return 0;
    for (int i = 1; i <= N; i++) printf("%d\n", i);
    return 0;
}
